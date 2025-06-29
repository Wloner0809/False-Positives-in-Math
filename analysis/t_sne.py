import os
import warnings
from itertools import product
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "DejaVu Serif"


class MathReasoningDataset(Dataset):
    """Dataset for math reasoning problems"""

    def __init__(
        self,
        problems: List[str],
        solutions: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 6000,
    ):
        self.problems = problems
        self.solutions = solutions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        problem = self.problems[idx]
        solution = self.solutions[idx]
        label = self.labels[idx]

        # Combine problem and solution
        text = f"Problem:\n\n{problem}\n\nSolution:\n\n{solution}"

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
            "text": text,
            "problem": problem,
            "solution": solution,
        }


class HiddenStateExtractor:
    """Extract hidden states from transformer models with model parallelism support for large models"""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        use_model_parallel: bool = True,
        max_memory_per_gpu: str = "70GiB",
        offload_folder: Optional[str] = None,
        low_cpu_mem_usage: bool = True,
        torch_dtype: str = "auto",
    ):
        self.model_name = model_name
        self.device = device
        self.use_model_parallel = use_model_parallel
        self.max_memory_per_gpu = max_memory_per_gpu
        self.offload_folder = offload_folder
        self.low_cpu_mem_usage = low_cpu_mem_usage

        self.torch_dtype = torch_dtype
        print(f"Using dtype: {self.torch_dtype}")

        # Load tokenizer first
        print(f"Loading tokenizer from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Check available GPUs
        self.num_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {self.num_gpus}")

        if self.num_gpus == 0:
            raise ValueError("No CUDA devices available!")

        # Load model with automatic device mapping for large models
        self.model = self._load_model_with_parallelism()
        self.model.eval()

        # Store device map for reference
        if hasattr(self.model, "hf_device_map"):
            self.device_map = self.model.hf_device_map
            print(f"Model device map: {self.device_map}")
        else:
            self.device_map = None

        # Print final memory usage
        print("GPU memory usage after model loading:")
        self._print_memory_usage()

    def _load_model_with_parallelism(self):
        """Load model with automatic device mapping for model parallelism"""
        if not self.use_model_parallel or self.num_gpus == 1:
            # Single GPU loading
            print(f"Loading model on single device: {self.device}")
            model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=self.low_cpu_mem_usage,
                trust_remote_code=True,
            )
            return model.to(self.device)

        # Multi-GPU model parallelism
        print(f"Loading model with model parallelism across {self.num_gpus} GPUs...")

        # Create max memory dict for each GPU
        max_memory = {}
        for i in range(self.num_gpus):
            max_memory[i] = self.max_memory_per_gpu

        # Add CPU memory if offloading is enabled
        if self.offload_folder:
            max_memory["cpu"] = "10GiB"  # Reserve some CPU memory for offloading

        print(f"Max memory configuration: {max_memory}")

        try:
            # Method 1: Use accelerate for automatic device mapping
            model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=self.low_cpu_mem_usage,
                device_map="auto",
                max_memory=max_memory,
                offload_folder=self.offload_folder,
                offload_state_dict=True if self.offload_folder else False,
                trust_remote_code=True,
            )
            print("Successfully loaded model with accelerate device_map='auto'")
            return model

        except Exception as e:
            print(f"Failed to load with accelerate auto device map: {e}")
            print(
                "This might be due to missing accelerate library or incompatible model"
            )

            try:
                # Method 2: Manual device mapping with better architecture detection
                print("Trying manual device mapping...")

                # Load config to determine model architecture
                config = AutoConfig.from_pretrained(
                    self.model_name, trust_remote_code=True
                )

                # Get the correct layer attribute name for different model architectures
                layer_attr = self._get_layer_attribute_name(config)
                num_layers = getattr(config, layer_attr, config.num_hidden_layers)

                print(f"Model has {num_layers} layers (using attribute: {layer_attr})")

                # Calculate layers per GPU
                layers_per_gpu = num_layers // self.num_gpus
                remainder = num_layers % self.num_gpus

                # Create manual device map based on model architecture
                device_map = self._create_manual_device_map(
                    config, num_layers, layers_per_gpu, remainder
                )

                print(f"Manual device map created: {device_map}")

                model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype,
                    low_cpu_mem_usage=self.low_cpu_mem_usage,
                    device_map=device_map,
                    max_memory=max_memory,
                    offload_folder=self.offload_folder,
                    trust_remote_code=True,
                )
                print("Successfully loaded model with manual device mapping")
                return model

            except Exception as e2:
                print(f"Failed to load with manual device mapping: {e2}")

                # Method 3: Fallback to single GPU with memory optimization
                print("Falling back to single GPU with memory optimization...")
                torch.cuda.empty_cache()  # Clear GPU cache

                try:
                    model = AutoModel.from_pretrained(
                        self.model_name,
                        torch_dtype=self.torch_dtype,
                        low_cpu_mem_usage=self.low_cpu_mem_usage,
                        trust_remote_code=True,
                    )
                    return model.to(self.device)
                except Exception as e3:
                    print(f"Failed to load on single GPU: {e3}")
                    raise RuntimeError(
                        f"Failed to load model with all methods. Final error: {e3}"
                    )

    def _get_layer_attribute_name(self, config):
        """Get the correct attribute name for the number of layers based on model architecture"""
        # Different model architectures use different attribute names
        possible_attrs = [
            "num_hidden_layers",  # Most common
            "n_layer",  # GPT-style models
            "num_layers",  # Some other models
            "n_layers",  # Alternative naming
        ]

        for attr in possible_attrs:
            if hasattr(config, attr):
                return attr

        # Fallback
        return "num_hidden_layers"

    def _create_manual_device_map(self, config, num_layers, layers_per_gpu, remainder):
        """Create manual device map based on model architecture"""
        device_map = {}

        # Common embedding layer names for different architectures
        embedding_names = [
            "embeddings",
            "embed_tokens",
            "wte",
            "word_embeddings",
            "transformer.wte",
            "model.embed_tokens",
        ]

        # Put embeddings on first GPU
        for embed_name in embedding_names:
            device_map[embed_name] = 0

        # Distribute transformer layers
        current_layer = 0

        # Different architectures use different layer naming conventions
        layer_patterns = [
            "layers.{}",
            "h.{}",
            "transformer.h.{}",
            "model.layers.{}",
            "decoder.layers.{}",
        ]

        for gpu_id in range(self.num_gpus):
            # Calculate how many layers for this GPU
            layers_for_this_gpu = layers_per_gpu + (1 if gpu_id < remainder else 0)

            # Assign layers to this GPU
            for _ in range(layers_for_this_gpu):
                if current_layer < num_layers:
                    # Try different layer naming patterns
                    for pattern in layer_patterns:
                        layer_name = pattern.format(current_layer)
                        device_map[layer_name] = gpu_id
                    current_layer += 1

        # Put final layers on last GPU
        final_layer_names = [
            "norm",
            "ln_f",
            "layer_norm",
            "final_layer_norm",
            "transformer.ln_f",
            "model.norm",
        ]

        for final_name in final_layer_names:
            device_map[final_name] = self.num_gpus - 1

        return device_map

    def _print_memory_usage(self):
        """Print current GPU memory usage"""
        for i in range(self.num_gpus):
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(
                    f"  GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total"
                )

    def get_memory_usage(self):
        """Get current GPU memory usage"""
        memory_info = {}
        for i in range(self.num_gpus):
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                memory_info[f"gpu_{i}"] = {
                    "allocated": f"{allocated:.2f} GB",
                    "reserved": f"{reserved:.2f} GB",
                    "total": f"{total:.2f} GB",
                    "utilization": f"{(allocated / total) * 100:.1f}%",
                }
        return memory_info

    def _move_batch_to_model_device(self, batch):
        """Move batch to appropriate device(s) based on model's device map"""
        if self.device_map is None:
            # Single device
            return {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

        # For model parallel, move to the device where embeddings are located
        embed_device = None
        for module_name, device in self.device_map.items():
            if "embed" in module_name.lower():
                embed_device = device
                break

        if embed_device is None:
            embed_device = 0  # Default to first GPU

        device_str = (
            f"cuda:{embed_device}" if isinstance(embed_device, int) else embed_device
        )

        return {
            k: v.to(device_str) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def extract_layer_representations(
        self,
        dataloader: DataLoader,
        layers: List[int] = None,
        pooling_strategy: str = "mean",
    ) -> Dict:
        """
        Extract hidden states from specified layers with model parallelism support
        Args:
            dataloader: DataLoader with math problems
            layers: List of layer indices to extract (None for all layers)
            pooling_strategy: 'mean', 'max', 'last_token', or 'cls'
        """
        # Get model configuration
        config = self.model.config
        if layers is None:
            layers = list(range(config.num_hidden_layers))

        all_hidden_states = {layer: [] for layer in layers}
        all_labels = []
        all_texts = []
        all_problems = []
        all_solutions = []

        # Print memory usage before starting
        print("Memory usage before extraction:")
        memory_info = self.get_memory_usage()
        for gpu, info in memory_info.items():
            print(f"  {gpu}: {info}")

        with torch.no_grad():
            progress_bar = tqdm(
                dataloader, desc="Extracting hidden states with model parallelism"
            )

            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to appropriate device(s)
                batch_tensors = self._move_batch_to_model_device(batch)

                input_ids = batch_tensors["input_ids"]
                attention_mask = batch_tensors["attention_mask"]
                labels = batch["label"]

                try:
                    # Forward pass with model parallelism
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )

                    hidden_states = outputs.hidden_states

                    for layer in layers:
                        layer_hidden = hidden_states[layer]

                        # Move attention mask to the same device as layer hidden states
                        if layer_hidden.device != attention_mask.device:
                            attention_mask_device = attention_mask.to(
                                layer_hidden.device
                            )
                        else:
                            attention_mask_device = attention_mask

                        # Apply pooling strategy
                        if pooling_strategy == "mean":
                            mask_expanded = (
                                attention_mask_device.unsqueeze(-1)
                                .expand(layer_hidden.size())
                                .float()
                            )
                            sum_hidden = torch.sum(layer_hidden * mask_expanded, dim=1)
                            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                            pooled_hidden = sum_hidden / sum_mask
                        elif pooling_strategy == "max":
                            pooled_hidden = torch.max(layer_hidden, dim=1)[0]
                        elif pooling_strategy == "last_token":
                            seq_lengths = attention_mask_device.sum(dim=1) - 1
                            pooled_hidden = layer_hidden[
                                range(layer_hidden.size(0)), seq_lengths
                            ]
                        elif pooling_strategy == "cls":
                            pooled_hidden = layer_hidden[:, 0, :]
                        else:
                            raise ValueError(
                                f"Unknown pooling strategy: {pooling_strategy}"
                            )

                        # Convert to float32 and move to CPU to avoid BFloat16 issues
                        all_hidden_states[layer].append(pooled_hidden.float().cpu())

                    all_labels.append(labels)
                    all_texts.extend(batch["text"])
                    all_problems.extend(batch["problem"])
                    all_solutions.extend(batch["solution"])

                    # Clear GPU cache periodically to prevent OOM
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()

                except torch.cuda.OutOfMemoryError as e:
                    print(f"CUDA OOM error at batch {batch_idx}: {e}")
                    torch.cuda.empty_cache()
                    # Try to continue with smaller batch or skip this batch
                    continue
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    continue

        # Concatenate all batches
        print("Concatenating results...")
        for layer in layers:
            if all_hidden_states[layer]:
                all_hidden_states[layer] = torch.cat(all_hidden_states[layer], dim=0)
            else:
                print(f"Warning: No data collected for layer {layer}")
                all_hidden_states[layer] = torch.empty(0)

        all_labels = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0)

        # Print final memory usage
        print("Memory usage after extraction:")
        memory_info = self.get_memory_usage()
        for gpu, info in memory_info.items():
            print(f"  {gpu}: {info}")

        return {
            "hidden_states": all_hidden_states,
            "labels": all_labels.numpy(),
            "texts": all_texts,
            "problems": all_problems,
            "solutions": all_solutions,
        }

    def extract_trajectory_representations(
        self,
        dataloader: DataLoader,
        target_layers: List[int] = None,
        pooling_strategy: str = "mean",
    ) -> Dict:
        """
        Extract hidden states trajectory across multiple layers for trajectory analysis
        with model parallelism support
        """
        if target_layers is None:
            target_layers = [-4, -3, -2, -1]  # Last 4 layers

        trajectories = []
        labels = []
        texts = []

        print("Memory usage before trajectory extraction:")
        memory_info = self.get_memory_usage()
        for gpu, info in memory_info.items():
            print(f"  {gpu}: {info}")

        with torch.no_grad():
            progress_bar = tqdm(
                dataloader, desc="Extracting trajectories with model parallelism"
            )

            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to appropriate device(s)
                batch_tensors = self._move_batch_to_model_device(batch)

                input_ids = batch_tensors["input_ids"]
                attention_mask = batch_tensors["attention_mask"]
                batch_labels = batch["label"]

                try:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )

                    hidden_states = outputs.hidden_states
                    batch_size = input_ids.size(0)

                    for i in range(batch_size):
                        trajectory = []
                        mask = attention_mask[i]

                        for layer_idx in target_layers:
                            layer_hidden = hidden_states[layer_idx][i]

                            # Move mask to the same device as layer hidden states if needed
                            if layer_hidden.device != mask.device:
                                mask_device = mask.to(layer_hidden.device)
                            else:
                                mask_device = mask

                            # Pooling for this sample
                            valid_positions = mask_device.bool()
                            if valid_positions.sum() > 0:
                                if pooling_strategy == "mean":
                                    pooled = layer_hidden[valid_positions].mean(dim=0)
                                elif pooling_strategy == "max":
                                    pooled = layer_hidden[valid_positions].max(dim=0)[0]
                                elif pooling_strategy == "last_token":
                                    last_token_idx = mask_device.nonzero(as_tuple=True)[
                                        0
                                    ][-1]
                                    pooled = layer_hidden[last_token_idx]
                                elif pooling_strategy == "cls":
                                    pooled = layer_hidden[0]
                            else:
                                pooled = layer_hidden[0]

                            # Convert to float32 and then to numpy to avoid BFloat16 issues
                            trajectory.append(pooled.float().cpu().numpy())

                        trajectories.append(np.array(trajectory))
                        labels.append(batch_labels[i].item())
                        texts.append(batch["text"][i])

                    # Clear GPU cache periodically
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()

                except torch.cuda.OutOfMemoryError as e:
                    print(f"CUDA OOM error at batch {batch_idx}: {e}")
                    torch.cuda.empty_cache()
                    continue
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    continue

        print("Memory usage after trajectory extraction:")
        memory_info = self.get_memory_usage()
        for gpu, info in memory_info.items():
            print(f"  {gpu}: {info}")

        return {
            "trajectories": trajectories,
            "labels": np.array(labels),
            "texts": texts,
            "target_layers": target_layers,
        }


class TSNEAnalyzer:
    """T-SNE analysis for hidden states visualization"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.label_names = {0: "Correct", 1: "False Positive", 2: "Incorrect"}
        self.label_colors = {0: "#2E8B57", 1: "#FF6B6B", 2: "#4169E1"}

    def perform_tsne(
        self,
        features: np.ndarray,
        perplexity: float = 30,
        n_iter: int = 1000,
        learning_rate: float = 200,
        early_exaggeration: float = 12.0,
        min_grad_norm: float = 1e-7,
        method: str = "barnes_hut",
        angle: float = 0.5,
    ) -> np.ndarray:
        """
        Perform T-SNE dimensionality reduction with enhanced parameters
        """
        # Ensure perplexity is not too large for the dataset
        max_perplexity = (len(features) - 1) // 3
        actual_perplexity = min(perplexity, max_perplexity)

        if actual_perplexity != perplexity:
            print(
                f"Warning: Perplexity reduced from {perplexity} to {actual_perplexity} due to dataset size"
            )

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Apply T-SNE with enhanced parameters
        tsne = TSNE(
            n_components=2,
            perplexity=actual_perplexity,
            n_iter_without_progress=n_iter,
            learning_rate=learning_rate,
            early_exaggeration=early_exaggeration,
            min_grad_norm=min_grad_norm,
            random_state=self.random_state,
            verbose=1,
            method=method,
            angle=angle,
        )

        tsne_results = tsne.fit_transform(features_scaled)
        return tsne_results

    def advanced_preprocessing(
        self,
        features: np.ndarray,
        method: str = "standard",
    ) -> np.ndarray:
        """
        Advanced data preprocessing methods
        """
        if method == "standard":
            scaler = StandardScaler()
            return scaler.fit_transform(features)

        elif method == "robust":
            from sklearn.preprocessing import RobustScaler

            scaler = RobustScaler()
            return scaler.fit_transform(features)

        elif method == "minmax":
            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler()
            return scaler.fit_transform(features)

        elif method == "pca_then_scale":
            from sklearn.decomposition import PCA

            # Apply PCA first, then standardize
            n_components = min(50, features.shape[1])
            pca = PCA(n_components=n_components)
            features_pca = pca.fit_transform(features)
            scaler = StandardScaler()
            return scaler.fit_transform(features_pca)

        else:
            raise ValueError(f"Unknown preprocessing method: {method}")

    def optimize_tsne_parameters(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        param_grid: Dict = None,
        scoring_metric: str = "silhouette",
    ) -> Dict:
        """
        Automatically optimize T-SNE parameters using grid search
        Args:
            features: Input features
            labels: Ground truth labels for scoring
            param_grid: Parameter grid to search
            scoring_metric: 'silhouette' or 'calinski_harabasz'
        """
        if param_grid is None:
            # Adaptive parameter grid based on dataset size
            n_samples = len(features)

            if n_samples < 100:
                param_grid = {
                    "perplexity": [5, 10, 15, 30],
                    "learning_rate": [50, 100, 150, 200],
                    "early_exaggeration": [4.0, 8.0, 12.0],
                    "n_iter": [1000, 1500, 2000],
                }
            elif n_samples < 1000:
                param_grid = {
                    "perplexity": [10, 20, 30, 40, 50],
                    "learning_rate": [100, 200, 300, 400, 500],
                    "early_exaggeration": [8.0, 12.0, 16.0],
                    "n_iter": [1000, 2000, 3000],
                }
            else:
                param_grid = {
                    "perplexity": [25, 50, 75, 100],
                    "learning_rate": [200, 400, 600, 800, 1000],
                    "early_exaggeration": [12.0, 16.0, 24.0],
                    "n_iter": [1500, 3000, 4500],
                    "method": ["barnes_hut"],
                }

        best_score = -1
        best_params = {}
        best_results = None

        # Use grid search

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        print(
            f"Starting parameter optimization with {len(list(product(*param_values)))} combinations..."
        )

        for params in tqdm(product(*param_values), desc="Optimizing parameters"):
            param_dict = dict(zip(param_names, params))

            try:
                # Perform T-SNE with current parameters
                tsne_results = self.perform_tsne(features=features, **param_dict)

                # Calculate score
                if scoring_metric == "silhouette":
                    if len(np.unique(labels)) > 1:  # Need at least 2 clusters
                        score = silhouette_score(tsne_results, labels)
                    else:
                        score = -1
                elif scoring_metric == "calinski_harabasz":
                    from sklearn.metrics import calinski_harabasz_score

                    score = calinski_harabasz_score(tsne_results, labels)
                else:
                    raise ValueError(f"Unknown scoring metric: {scoring_metric}")

                print(f"Params: {param_dict}, Score: {score:.3f}")

                if score > best_score:
                    best_score = score
                    best_params = param_dict
                    best_results = tsne_results

            except Exception as e:
                print(f"Failed with params {param_dict}: {e}")
                continue

        return {
            "best_params": best_params,
            "best_score": best_score,
            "best_results": best_results,
            "scoring_metric": scoring_metric,
        }

    def diagnose_clustering_issues(
        self, features: np.ndarray, labels: np.ndarray
    ) -> Dict:
        """
        Diagnose clustering issues and provide recommendations
        """
        n_samples, n_features = features.shape
        n_classes = len(np.unique(labels))

        recommendations = []

        # Check dataset size
        if n_samples < 50:
            recommendations.append("Dataset too small: recommend perplexity=5-10")
        elif n_samples > 10000:
            recommendations.append(
                "Large dataset: recommend method='barnes_hut', increase learning_rate"
            )

        # Check feature dimensions
        if n_features > 1000:
            recommendations.append(
                "High-dimensional features: recommend PCA preprocessing to 50-100 dimensions"
            )

        # Check class balance
        unique_labels, counts = np.unique(labels, return_counts=True)
        class_counts = dict(zip(unique_labels, counts))
        max_count = np.max(counts)
        min_count = np.min(counts)

        if max_count / min_count > 10:
            recommendations.append(
                "Severe class imbalance: consider balanced sampling or visualization strategies"
            )

        # Check feature variance
        feature_var = np.var(features, axis=0)
        if np.max(feature_var) / np.min(feature_var) > 1000:
            recommendations.append(
                "High feature variance difference: recommend StandardScaler or RobustScaler"
            )

        return {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": n_classes,
            "class_distribution": class_counts,
            "feature_variance_ratio": (
                np.max(feature_var) / np.min(feature_var)
                if np.min(feature_var) > 0
                else float("inf")
            ),
            "recommendations": recommendations,
        }

    def comprehensive_parameter_search(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> Dict:
        """
        Comprehensive parameter search with multiple strategies
        """
        print("=== Starting Comprehensive Parameter Search ===")

        # 1. Diagnose issues
        diagnosis = self.diagnose_clustering_issues(features, labels)
        print("\n1. Dataset Diagnosis:")
        for rec in diagnosis["recommendations"]:
            print(f"   - {rec}")

        results = {}

        # 2. Test different preprocessing methods
        print("\n2. Testing preprocessing methods...")
        preprocessing_methods = ["standard", "robust", "minmax"]
        if features.shape[1] > 100:
            preprocessing_methods.append("pca_then_scale")

        best_preprocessing = "standard"
        best_preprocessing_score = -1

        for method in preprocessing_methods:
            try:
                processed_features = self.advanced_preprocessing(features, method)

                # Quick T-SNE test
                tsne_result = self.perform_tsne(
                    processed_features,
                    perplexity=min(30, (len(features) - 1) // 3),
                    n_iter=2000,
                )

                if len(np.unique(labels)) > 1:
                    score = silhouette_score(tsne_result, labels)
                    print(f"   {method}: {score:.3f}")

                    if score > best_preprocessing_score:
                        best_preprocessing_score = score
                        best_preprocessing = method

            except Exception as e:
                print(f"   {method}: Failed - {e}")

        print(f"   Best preprocessing: {best_preprocessing}")

        # 3. Optimize parameters with best preprocessing
        print("\n3. Optimizing T-SNE parameters...")
        processed_features = self.advanced_preprocessing(features, best_preprocessing)

        optimization_result = self.optimize_tsne_parameters(processed_features, labels)

        results.update(
            {
                "diagnosis": diagnosis,
                "best_preprocessing": best_preprocessing,
                "optimization_result": optimization_result,
            }
        )

        return results

    def visualize_parameter_comparison(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize T-SNE results with different parameter settings for comparison
        """
        import matplotlib.pyplot as plt

        # Test different perplexity values
        perplexities = [5, 15, 30, 50]
        max_perplexity = (len(features) - 1) // 3
        perplexities = [p for p in perplexities if p <= max_perplexity]

        fig, axes = plt.subplots(2, len(perplexities), figsize=(20, 12))
        if len(perplexities) == 1:
            axes = axes.reshape(2, 1)

        # Standardize features once
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Test different perplexilities
        for i, perp in enumerate(perplexities):
            tsne = TSNE(
                n_components=2,
                perplexity=perp,
                n_iter_without_progress=2000,
                learning_rate=200,
                random_state=self.random_state,
                verbose=0,
            )

            tsne_results = tsne.fit_transform(features_scaled)

            # Calculate silhouette score
            if len(np.unique(labels)) > 1:
                sil_score = silhouette_score(tsne_results, labels)
            else:
                sil_score = 0.0

            # Plot in first row
            for label in np.unique(labels):
                mask = labels == label
                axes[0, i].scatter(
                    tsne_results[mask, 0],
                    tsne_results[mask, 1],
                    c=self.label_colors[label],
                    label=self.label_names[label],
                    alpha=0.7,
                    s=30,
                )

            axes[0, i].set_title(
                f"Perplexity={perp} (LR=200)\nSilhouette={sil_score:.3f}"
            )
            axes[0, i].legend(fontsize=8)
            axes[0, i].grid(True, alpha=0.3)

        # Test different learning rates
        learning_rates = [50, 100, 200]
        optimal_perp = perplexities[len(perplexities) // 2] if perplexities else 15

        for i, lr in enumerate(learning_rates):
            if i >= len(perplexities):
                break

            tsne = TSNE(
                n_components=2,
                perplexity=optimal_perp,
                n_iter_without_progress=2000,
                learning_rate=lr,
                random_state=self.random_state,
                verbose=0,
            )

            tsne_results = tsne.fit_transform(features_scaled)

            # Calculate silhouette score
            if len(np.unique(labels)) > 1:
                sil_score = silhouette_score(tsne_results, labels)
            else:
                sil_score = 0.0

            # Plot in second row
            for label in np.unique(labels):
                mask = labels == label
                axes[1, i].scatter(
                    tsne_results[mask, 0],
                    tsne_results[mask, 1],
                    c=self.label_colors[label],
                    label=self.label_names[label],
                    alpha=0.7,
                    s=30,
                )

            axes[1, i].set_title(
                f"LR={lr} (perp={optimal_perp})\nSilhouette={sil_score:.3f}"
            )
            axes[1, i].legend(fontsize=8)
            axes[1, i].grid(True, alpha=0.3)

        # Fill remaining subplots if needed
        for i in range(len(learning_rates), len(perplexities)):
            axes[1, i].axis("off")

        plt.suptitle("T-SNE Parameter Comparison", fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def visualize_clustering(
        self,
        tsne_results: np.ndarray,
        labels: np.ndarray,
        title: str = "T-SNE Clustering",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize T-SNE results with clustering
        """
        plt.figure(figsize=(10, 8))

        for label in np.unique(labels):
            mask = labels == label
            plt.scatter(
                tsne_results[mask, 0],
                tsne_results[mask, 1],
                c=self.label_colors[label],
                label=self.label_names[label],
                alpha=0.7,
                s=30,
            )

        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def visualize_trajectory_analysis(
        self,
        trajectories: List[np.ndarray],
        labels: np.ndarray,
        layer_names: List[str],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize reasoning trajectory analysis
        """
        # Check if we have any trajectories
        if not trajectories or len(trajectories) == 0:
            print("Warning: No trajectories available for visualization")
            return

        # Flatten trajectories for T-SNE
        flattened_trajectories = []
        for traj in trajectories:
            flattened_trajectories.append(traj.flatten())

        trajectory_features = np.array(flattened_trajectories)

        # Check if trajectory_features is empty or has invalid shape
        if trajectory_features.size == 0:
            print("Warning: Empty trajectory features, skipping visualization")
            return

        if len(trajectory_features.shape) != 2 or trajectory_features.shape[0] == 0:
            print(
                f"Warning: Invalid trajectory features shape {trajectory_features.shape}, skipping visualization"
            )
            return

        # Apply T-SNE to trajectories
        tsne_trajectories = self.perform_tsne(trajectory_features, perplexity=15)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        # 1. Trajectory T-SNE
        for label in np.unique(labels):
            mask = labels == label
            axes[0, 0].scatter(
                tsne_trajectories[mask, 0],
                tsne_trajectories[mask, 1],
                c=self.label_colors[label],
                label=self.label_names[label],
                alpha=0.7,
                s=50,
            )
        axes[0, 0].set_title("Trajectory T-SNE")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Layer-wise progression for each class
        for label in [0, 1, 2]:
            mask = labels == label
            if mask.sum() == 0:
                continue

            class_trajectories = [
                trajectories[i] for i in range(len(trajectories)) if mask[i]
            ]

            # Average trajectory for this class
            avg_trajectory = np.mean(class_trajectories, axis=0)

            # Apply T-SNE to each layer's representations
            layer_tsne_results = []
            for layer_idx in range(avg_trajectory.shape[0]):
                layer_features = np.array(
                    [traj[layer_idx] for traj in class_trajectories]
                )
                if len(layer_features) > 1:
                    layer_tsne = self.perform_tsne(
                        layer_features, perplexity=min(5, len(layer_features) - 1)
                    )
                    layer_tsne_results.append(layer_tsne)

            if layer_tsne_results:
                ax_idx = (0, 1) if label == 0 else ((1, 0) if label == 1 else (1, 1))
                ax = axes[ax_idx]

                for i, layer_tsne in enumerate(layer_tsne_results):
                    ax.scatter(
                        layer_tsne[:, 0],
                        layer_tsne[:, 1],
                        label=f"Layer {layer_names[i]}",
                        alpha=0.6,
                    )

                ax.set_title(f"{self.label_names[label]} - Layer Progression")
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def analyze_cluster_separation(
        self, tsne_results: np.ndarray, labels: np.ndarray
    ) -> Dict:
        """
        Analyze cluster separation using silhouette score and other metrics
        """
        from sklearn.metrics import calinski_harabasz_score

        # Calculate silhouette score
        if len(np.unique(labels)) > 1:
            sil_score = silhouette_score(tsne_results, labels)
        else:
            sil_score = 0.0

        # Calculate Calinski-Harabasz score
        ch_score = calinski_harabasz_score(tsne_results, labels)

        return {
            "silhouette_score": sil_score,
            "calinski_harabasz_score": ch_score,
        }

    def interactive_visualization(
        self,
        tsne_results: np.ndarray,
        labels: np.ndarray,
        texts: List[str],
        title: str = "T-SNE Visualization",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create an enhanced interactive scatter plot for T-SNE results with beautiful styling
        """

        # Create DataFrame with enhanced data
        df_tsne = pd.DataFrame(tsne_results, columns=["x", "y"])
        df_tsne["label"] = [self.label_names[label] for label in labels]
        df_tsne["label_numeric"] = labels
        df_tsne["text"] = texts

        # Truncate text for better hover display
        df_tsne["text_preview"] = [
            text[:200] + "..." if len(text) > 200 else text for text in texts
        ]

        # Add sample index for reference
        df_tsne["sample_id"] = range(len(df_tsne))

        # Enhanced color palette with better contrast
        enhanced_colors = {
            "Correct": "#2E8B57",
            "False Positive": "#FF6B6B",
            "Incorrect": "#4169E1",
        }

        # Create the main scatter plot
        fig = px.scatter(
            df_tsne,
            x="x",
            y="y",
            color="label",
            hover_data={
                "x": ":.2f",
                "y": ":.2f",
                "text_preview": True,
                "sample_id": True,
                "label": False,
            },
            title=title,
            color_discrete_map=enhanced_colors,
            width=1000,
            height=700,
        )

        # Enhanced marker styling with size variation based on label
        label_sizes = {"Correct": 8, "False Positive": 10, "Incorrect": 8}

        for trace in fig.data:
            label_name = trace.name
            fig.update_traces(
                marker=dict(
                    size=label_sizes.get(label_name, 10),
                    opacity=0.8,
                    line=dict(width=1.5, color="white"),
                    symbol="circle",
                ),
                selector=dict(name=label_name),
            )

        # Enhanced layout with beautiful styling
        fig.update_layout(
            # Title styling
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                xanchor="center",
                font=dict(size=20, family="Arial Black", color="#2c3e50"),
            ),
            # Background and grid
            plot_bgcolor="rgba(248, 249, 250, 0.95)",
            paper_bgcolor="white",
            # Axes styling
            xaxis=dict(
                title=dict(
                    text="<b>t-SNE Dimension 1</b>",
                    font=dict(size=14, family="Arial", color="#34495e"),
                ),
                gridcolor="rgba(0,0,0,0.1)",
                gridwidth=1,
                zeroline=True,
                zerolinecolor="rgba(0,0,0,0.3)",
                zerolinewidth=2,
                showline=True,
                linecolor="rgba(0,0,0,0.3)",
                linewidth=1,
                tickfont=dict(size=12, color="#2c3e50"),
            ),
            yaxis=dict(
                title=dict(
                    text="<b>t-SNE Dimension 2</b>",
                    font=dict(size=14, family="Arial", color="#34495e"),
                ),
                gridcolor="rgba(0,0,0,0.1)",
                gridwidth=1,
                zeroline=True,
                zerolinecolor="rgba(0,0,0,0.3)",
                zerolinewidth=2,
                showline=True,
                linecolor="rgba(0,0,0,0.3)",
                linewidth=1,
                tickfont=dict(size=12, color="#2c3e50"),
            ),
            # Enhanced legend
            legend=dict(
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.15,
                yanchor="top",
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
                font=dict(size=12, family="Arial", color="#2c3e50"),
                itemsizing="constant",
                title=dict(
                    text="<b>Sample Categories</b>",
                    font=dict(size=14, family="Arial", color="#2c3e50"),
                ),
            ),
            # Margins and spacing
            margin=dict(l=50, r=50, t=80, b=100),
            # Hover styling
            hoverlabel=dict(
                bgcolor="rgba(255, 255, 255, 0.95)",
                bordercolor="rgba(0,0,0,0.2)",
                font=dict(size=11, family="Arial", color="#2c3e50"),
                align="left",
            ),
            # Add subtle shadow effect
            annotations=[
                dict(
                    text="",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0,
                    y=0,
                    xanchor="left",
                    yanchor="bottom",
                    xshift=-5,
                    yshift=-5,
                    bordercolor="rgba(0,0,0,0.1)",
                    borderwidth=3,
                    bgcolor="rgba(0,0,0,0.05)",
                    width=1000,
                    height=700,
                )
            ],
        )

        # Add custom hover template
        fig.update_traces(
            hovertemplate="<b>%{customdata[3]}</b><br>"
            + "Sample ID: %{customdata[2]}<br>"
            + "Position: (%{x:.2f}, %{y:.2f})<br>"
            + "<b>Text Preview:</b><br>%{customdata[1]}"
            + "<extra></extra>",
            customdata=df_tsne[
                ["label_numeric", "text_preview", "sample_id", "label"]
            ].values,
        )

        # Enhanced save and display
        if save_path:
            # Save with additional configuration for better web display
            fig.write_html(
                save_path,
                config={
                    "displayModeBar": True,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "tsne_visualization",
                        "height": 700,
                        "width": 1000,
                        "scale": 2,
                    },
                },
                div_id="tsne-plot",
                include_plotlyjs=True,
            )
            print(f"Enhanced interactive visualization saved to: {save_path}")

        fig.show(config={"displayModeBar": True, "displaylogo": False})


def load_sample_data(data_path: str = "sample_math_data.jsonl") -> tuple:
    """Load sample mathematical reasoning data"""
    df = pd.read_json(data_path, lines=True)
    problems = df["problem"].tolist()
    solutions = df["solution"].tolist()
    labels = df["label"].tolist()

    return problems, solutions, labels


def create_simple_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 2,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a simple dataloader optimized for model parallelism"""
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep consistent order for analysis
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def main_tsne_analysis(
    model_name_or_path: str = None,
    use_model_parallel: bool = True,
    max_memory_per_gpu: str = "70GiB",
    offload_folder: str = None,
    batch_size: int = 8,
    torch_dtype: str = "bfloat16",
    data_path: str = "results/analysis/correct_false_positive_incorrect_qwen.jsonl",
    device: str = "cuda",
    max_length: int = 6000,
    target_layers: List[int] = [-4, -3, -2, -1],
    pooling_strategy: str = "last_token",
    save_dir: str = None,
):
    """Main T-SNE analysis pipeline with model parallelism support for large models"""
    # Create save directory based on model name
    model_name = model_name_or_path.split("/")[-1]

    print(f"=== T-SNE Analysis for {model_name} ===")
    print(f"Model path: {model_name_or_path}")
    print(f"Data path: {data_path}")
    print(f"Save directory: {save_dir}")
    print(f"Max memory per GPU: {max_memory_per_gpu}")
    print(f"Batch size: {batch_size}")
    print(f"Max length: {max_length}")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Load sample data
    print("\nLoading data...")
    problems, solutions, labels = load_sample_data(data_path=data_path)
    print(f"Loaded {len(problems)} samples")

    # Setup tokenizer separately (to avoid loading it multiple times)
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Create dataset
    dataset = MathReasoningDataset(
        problems=problems,
        solutions=solutions,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    # Create dataloader (no distributed sampling for model parallel)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep consistent order for analysis
        num_workers=2,  # Reduce workers to save memory
        pin_memory=False,  # Disable to save memory
        drop_last=False,
    )

    # Initialize extractor with model parallelism
    print("\nInitializing model with parallelism...")
    extractor = HiddenStateExtractor(
        model_name=model_name_or_path,
        device=device,
        use_model_parallel=use_model_parallel,
        max_memory_per_gpu=max_memory_per_gpu,
        offload_folder=offload_folder,
        torch_dtype=torch_dtype,
    )

    # Initialize analyzer
    analyzer = TSNEAnalyzer()

    print("\n=== Starting T-SNE Analysis ===")
    print(f"Using {torch.cuda.device_count()} GPUs with model parallelism")

    # 1. Extract hidden states from multiple layers
    print("\n1. Extracting hidden states...")
    extracted_data = extractor.extract_layer_representations(
        dataloader=dataloader, layers=target_layers, pooling_strategy=pooling_strategy
    )

    print(f"Extracted hidden states for {len(extracted_data['labels'])} samples")

    # 2. Comprehensive T-SNE analysis with parameter optimization
    print("\n2. Performing comprehensive T-SNE analysis with parameter optimization...")
    layer_results = {}
    optimization_results = {}

    for layer in target_layers:
        print(f"\n=== Analyzing layer {layer} ===")

        if (
            layer not in extracted_data["hidden_states"]
            or len(extracted_data["hidden_states"][layer]) == 0
        ):
            print(f"Warning: No data for layer {layer}, skipping...")
            continue

        features = extracted_data["hidden_states"][layer].numpy()
        labels_array = extracted_data["labels"]

        print(f"Layer {layer} feature shape: {features.shape}")

        # Check if features are empty or invalid
        if features.size == 0:
            print(f"Warning: Empty features for layer {layer}, skipping...")
            continue

        if len(features.shape) != 2 or features.shape[0] == 0:
            print(
                f"Warning: Invalid feature shape {features.shape} for layer {layer}, skipping..."
            )
            continue

        # Comprehensive parameter search
        comprehensive_result = analyzer.comprehensive_parameter_search(
            features=features, labels=labels_array
        )

        optimization_results[layer] = comprehensive_result

        # Use optimized parameters for final T-SNE
        best_params = comprehensive_result["optimization_result"]["best_params"]
        best_preprocessing = comprehensive_result["best_preprocessing"]

        print(f"Best parameters for layer {layer}: {best_params}")
        print(f"Best preprocessing: {best_preprocessing}")
        print(
            f"Best score: {comprehensive_result['optimization_result']['best_score']:.3f}"
        )

        # Apply best preprocessing and T-SNE
        processed_features = analyzer.advanced_preprocessing(
            features, best_preprocessing
        )
        optimized_tsne_results = analyzer.perform_tsne(
            features=processed_features, **best_params
        )

        # Store results
        layer_results[layer] = {
            "tsne_results": optimized_tsne_results,
            "labels": labels_array,
            "features": features,
            "processed_features": processed_features,
            "best_params": best_params,
            "best_preprocessing": best_preprocessing,
            "optimization_score": comprehensive_result["optimization_result"][
                "best_score"
            ],
        }

        # Visualize optimized clustering
        analyzer.visualize_clustering(
            tsne_results=optimized_tsne_results,
            labels=labels_array,
            title=f"Layer {layer} Optimized T-SNE (Score: {comprehensive_result['optimization_result']['best_score']:.3f})",
            save_path=f"{save_dir}/layer_{layer}_optimized_tsne.png",
        )

        # Analyze cluster separation with optimized results
        separation_metrics = analyzer.analyze_cluster_separation(
            tsne_results=optimized_tsne_results, labels=labels_array
        )
        print(
            f"Layer {layer} Final Silhouette Score: {separation_metrics['silhouette_score']:.3f}"
        )

        # Save optimization details
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/layer_{layer}_optimization_details.txt", "w") as f:
            f.write(f"Layer {layer} Optimization Results\n")
            f.write("=" * 40 + "\n\n")
            f.write("Dataset Info:\n")
            f.write(f"  Samples: {comprehensive_result['diagnosis']['n_samples']}\n")
            f.write(f"  Features: {comprehensive_result['diagnosis']['n_features']}\n")
            f.write(f"  Classes: {comprehensive_result['diagnosis']['n_classes']}\n")
            f.write(
                f"  Class distribution: {comprehensive_result['diagnosis']['class_distribution']}\n\n"
            )
            f.write("Recommendations:\n")
            for rec in comprehensive_result["diagnosis"]["recommendations"]:
                f.write(f"  - {rec}\n")
            f.write(f"\nBest Preprocessing: {best_preprocessing}\n")
            f.write(f"Best Parameters: {best_params}\n")
            f.write(
                f"Best Score: {comprehensive_result['optimization_result']['best_score']:.3f}\n"
            )
            f.write(
                f"Final Silhouette Score: {separation_metrics['silhouette_score']:.3f}\n"
            )

        # Create parameter comparison visualization
        print(f"Creating parameter comparison for layer {layer}...")
        analyzer.visualize_parameter_comparison(
            features=processed_features,
            labels=labels_array,
            save_path=f"{save_dir}/layer_{layer}_parameter_comparison.png",
        )

    # 3. Trajectory analysis
    print("\n3. Performing trajectory analysis...")
    trajectory_data = extractor.extract_trajectory_representations(
        dataloader=dataloader,
        target_layers=target_layers,
        pooling_strategy=pooling_strategy,
    )

    # Check if we have valid trajectory data before visualization
    if trajectory_data["trajectories"] and len(trajectory_data["trajectories"]) > 0:
        analyzer.visualize_trajectory_analysis(
            trajectories=trajectory_data["trajectories"],
            labels=trajectory_data["labels"],
            layer_names=[f"Layer {layer}" for layer in target_layers],
            save_path=f"{save_dir}/trajectory_analysis.png",
        )
    else:
        print(
            "Warning: No valid trajectory data available, skipping trajectory visualization"
        )

    # 4. Interactive visualization for best layer
    if layer_results:  # Check if we have any layer results
        best_layer = max(
            layer_results.keys(),
            key=lambda layer: layer_results[layer]["optimization_score"],
        )

        print(f"\n4. Creating interactive visualization for best layer: {best_layer}")
        print(
            f"Best layer score: {layer_results[best_layer]['optimization_score']:.3f}"
        )
        analyzer.interactive_visualization(
            tsne_results=layer_results[best_layer]["tsne_results"],
            labels=layer_results[best_layer]["labels"],
            texts=extracted_data["texts"],
            title=f"T-SNE - Layer {best_layer}",
            save_path=f"{save_dir}/interactive_best_layer_{best_layer}.html",
        )
    else:
        print("Warning: No layer results available, skipping interactive visualization")
        best_layer = None

    # 5. Create summary report
    print("\n5. Creating optimization summary report...")
    with open(f"{save_dir}/tsne_optimization_summary.txt", "w") as f:
        f.write("T-SNE Optimization Summary Report\n")
        f.write("=" * 50 + "\n\n")

        if layer_results:
            for layer in layer_results.keys():
                result = layer_results[layer]
                f.write(f"Layer {layer}:\n")
                f.write(f"  Optimization Score: {result['optimization_score']:.3f}\n")
                f.write(f"  Best Preprocessing: {result['best_preprocessing']}\n")
                f.write(f"  Best Parameters: {result['best_params']}\n")
                f.write("\n")

            if best_layer is not None:
                f.write(f"Best Overall Layer: {best_layer}\n")
                f.write(
                    f"Best Overall Score: {layer_results[best_layer]['optimization_score']:.3f}\n"
                )
        else:
            f.write("No layer results available.\n")

    print("\n=== Analysis Complete ===")
    if best_layer is not None:
        print(f"Best performing layer: {best_layer}")
        print(f"Best score: {layer_results[best_layer]['optimization_score']:.3f}")
    else:
        print("No valid results obtained.")
    print(f"Results saved to: {save_dir}")

    return layer_results, trajectory_data, optimization_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="T-SNE Analysis with Model Parallelism for Large Models"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, default="Qwen/Qwen2.5-Math-72B"
    )
    parser.add_argument("--no_model_parallel", action="store_true")
    parser.add_argument("--max_memory_per_gpu", type=str, default="70GiB")
    parser.add_argument("--offload_folder", type=str, default="offload")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="results/analysis/correct_false_positive_incorrect_qwen.jsonl",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_length", type=int, default=6000)
    parser.add_argument(
        "--target_layers",
        type=int,
        nargs="+",
        default=[-1, -2, -3, -4],
        help="Layers to analyze (default: last 4 layers)",
    )
    parser.add_argument("--pooling_strategy", type=str, default="mean")
    parser.add_argument(
        "--save_dir", type=str, default="results/analysis/t_sne_results"
    )

    args = parser.parse_args()

    try:
        results = main_tsne_analysis(
            model_name_or_path=args.model_name_or_path,
            use_model_parallel=not args.no_model_parallel,
            max_memory_per_gpu=args.max_memory_per_gpu,
            offload_folder=args.offload_folder,
            batch_size=args.batch_size,
            torch_dtype=args.torch_dtype,
            data_path=args.data_path,
            device=args.device,
            max_length=args.max_length,
            target_layers=args.target_layers,
            pooling_strategy=args.pooling_strategy,
            save_dir=args.save_dir,
        )

        if results:
            print("\n=== Analysis Complete ===")
            print("Results saved successfully!")
        else:
            print("Analysis failed or was interrupted.")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()
