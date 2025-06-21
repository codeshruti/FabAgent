import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
import logging

class FabricBlendOptimizer:
    def __init__(self, materials_df: pd.DataFrame, max_materials: int = 5):
        """
        Initialize the fabric blend optimizer
        
        Args:
            materials_df: DataFrame with materials and their metrics
            max_materials: Maximum number of materials in a blend
        """
        self.materials_df = materials_df
        self.max_materials = max_materials
        self.material_names = materials_df['material'].tolist()
        self.n_materials = len(self.material_names)
        
        # Define metric categories
        self.environmental_metrics = [
            'water_consumption', 'ghg_emissions', 'land_use', 
            'pesticide_usage', 'biodegradation_time', 'energy_consumption'
        ]
        
        self.durability_metrics = [
            'tensile_strength', 'elongation_at_break', 'youngs_modulus',
            'abrasion_cycles', 'burst_strength', 'uv_resistance'
        ]
        
        self.comfort_metrics = [
            'moisture_regain', 'air_permeability', 'thermal_conductivity',
            'wicking_rate', 'static_resistance', 'uv_protection_factor'
        ]
        
        self.cost_metrics = [
            'raw_material_cost', 'processing_cost', 'dyeing_cost',
            'waste_percentage', 'energy_cost', 'total_manufacturing_cost'
        ]
        
        # Filter available metrics
        self.available_metrics = [col for col in materials_df.columns 
                                if col not in ['material', 'category']]
        
        self.environmental_metrics = [m for m in self.environmental_metrics 
                                    if m in self.available_metrics]
        self.durability_metrics = [m for m in self.durability_metrics 
                                 if m in self.available_metrics]
        self.comfort_metrics = [m for m in self.comfort_metrics 
                              if m in self.available_metrics]
        self.cost_metrics = [m for m in self.cost_metrics 
                           if m in self.available_metrics]
        
        # Normalize data
        self.normalized_df = self._normalize_data()
        
    def _normalize_data(self) -> pd.DataFrame:
        """Normalize the materials data using log transformation"""
        normalized_df = self.materials_df.copy()
        
        # Apply log transformation to reduce skewness
        for metric in self.available_metrics:
            if metric in normalized_df.columns:
                # Add small constant to avoid log(0)
                min_val = normalized_df[metric].min()
                if min_val <= 0:
                    normalized_df[metric] = normalized_df[metric] - min_val + 1
                
                # Apply log transformation
                normalized_df[metric] = np.log(normalized_df[metric])
        
        return normalized_df
    
    def _calculate_blend_score(self, blend_proportions: np.ndarray, 
                             metrics: List[str], maximize: bool = True) -> float:
        """
        Calculate the weighted average score for a blend
        
        Args:
            blend_proportions: Array of material proportions (sums to 1)
            metrics: List of metrics to consider
            maximize: Whether to maximize (True) or minimize (False) the score
            
        Returns:
            Blend score
        """
        if len(metrics) == 0:
            return 0.0
        
        # Calculate weighted average for each metric
        metric_scores = []
        for metric in metrics:
            if metric in self.normalized_df.columns:
                # Weighted average of the metric across all materials
                weighted_avg = np.sum(blend_proportions * self.normalized_df[metric].values)
                metric_scores.append(weighted_avg)
        
        if len(metric_scores) == 0:
            return 0.0
        
        # Average across all metrics
        avg_score = np.mean(metric_scores)
        
        # Return negative score if minimizing
        return avg_score if maximize else -avg_score
    
    def _repair_solution(self, x: np.ndarray) -> np.ndarray:
        """
        Repair solution to ensure only top K materials are used
        
        Args:
            x: Raw solution array
            
        Returns:
            Repaired solution array
        """
        # Add some randomness to avoid always selecting the same materials
        noise = np.random.normal(0, 0.1, len(x))
        x_with_noise = x + noise
        
        # Keep only the top K materials, but with some flexibility
        top_indices = np.argsort(x_with_noise)[-self.max_materials:]
        repaired = np.zeros_like(x)
        repaired[top_indices] = x[top_indices]
        
        # Normalize to sum to 1
        if float(np.sum(repaired)) > 0:
            repaired = repaired / np.sum(repaired)
        
        return repaired

class FabricBlendProblem(Problem):
    def __init__(self, optimizer: FabricBlendOptimizer):
        """
        Multi-objective optimization problem for fabric blends
        
        Objectives:
        1. Minimize environmental impact
        2. Minimize cost
        3. Maximize durability
        4. Maximize comfort
        """
        super().__init__(
            n_var=optimizer.n_materials,  # Number of materials
            n_obj=4,  # 4 objectives
            n_constr=1,  # 1 constraint (sum of proportions = 1)
            xl=0.0,  # Lower bound for each material proportion
            xu=1.0   # Upper bound for each material proportion
        )
        self.optimizer = optimizer
    
    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate the objectives and constraints for a population of solutions
        
        Args:
            x: Population of solutions (n_individuals x n_materials)
            out: Output dictionary to store results
        """
        n_individuals = x.shape[0]
        
        # Initialize objective and constraint arrays
        f = np.zeros((n_individuals, 4))
        g = np.zeros((n_individuals, 1))
        
        for i in range(n_individuals):
            # Repair solution to use only top K materials
            repaired_x = self.optimizer._repair_solution(x[i])
            
            # Calculate objectives
            # 1. Environmental impact (minimize)
            f[i, 0] = self.optimizer._calculate_blend_score(
                repaired_x, self.optimizer.environmental_metrics, maximize=False
            )
            
            # 2. Cost (minimize)
            f[i, 1] = self.optimizer._calculate_blend_score(
                repaired_x, self.optimizer.cost_metrics, maximize=False
            )
            
            # 3. Durability (maximize)
            f[i, 2] = self.optimizer._calculate_blend_score(
                repaired_x, self.optimizer.durability_metrics, maximize=True
            )
            
            # 4. Comfort (maximize)
            f[i, 3] = self.optimizer._calculate_blend_score(
                repaired_x, self.optimizer.comfort_metrics, maximize=True
            )
            
            # Constraint: sum of proportions should be close to 1
            g[i, 0] = np.abs(np.sum(repaired_x) - 1.0) - 0.01
        
        out["F"] = f
        out["G"] = g

def optimize_fabric_blend(materials_df: pd.DataFrame, 
                         max_materials: int = 5,
                         population_size: int = 100,
                         n_generations: int = 50) -> Dict[str, Any]:
    """
    Optimize fabric blend using NSGA-II algorithm
    
    Args:
        materials_df: DataFrame with materials and their metrics
        max_materials: Maximum number of materials in blend
        population_size: Size of the population
        n_generations: Number of generations
        
    Returns:
        Dictionary containing optimization results
    """
    # Initialize optimizer
    optimizer = FabricBlendOptimizer(materials_df, max_materials)
    
    # Create optimization problem
    problem = FabricBlendProblem(optimizer)
    
    # Initialize algorithm with different parameters based on input
    # Adjust algorithm parameters based on input to make it more sensitive to changes
    crossover_prob = 0.9 if population_size > 100 else 0.8
    mutation_prob = 1.0 / optimizer.n_materials  # Adaptive mutation probability
    
    algorithm = NSGA2(
        pop_size=population_size,
        sampling=LHS(),
        crossover=SBX(prob=crossover_prob, eta=15),
        mutation=PM(prob=mutation_prob, eta=20),
        eliminate_duplicates=True
    )
    
    # Use a seed based on the input parameters to make it deterministic but different for different inputs
    seed = hash(f"{max_materials}_{population_size}_{n_generations}") % 10000
    
    # Run optimization
    res = minimize(
        problem,
        algorithm,
        ('n_gen', n_generations),
        verbose=True,
        seed=seed
    )
    
    # Process results
    results = []
    for i, solution in enumerate(res.X):
        # Repair solution
        repaired_solution = optimizer._repair_solution(solution)
        
        # Get material proportions
        material_proportions = {}
        for j, material in enumerate(optimizer.material_names):
            if repaired_solution[j] > 0.01:  # Only include materials with >1% proportion
                material_proportions[material] = repaired_solution[j] * 100
        
        # Calculate scores
        environmental_score = optimizer._calculate_blend_score(
            repaired_solution, optimizer.environmental_metrics, maximize=False
        )
        cost_score = optimizer._calculate_blend_score(
            repaired_solution, optimizer.cost_metrics, maximize=False
        )
        durability_score = optimizer._calculate_blend_score(
            repaired_solution, optimizer.durability_metrics, maximize=True
        )
        comfort_score = optimizer._calculate_blend_score(
            repaired_solution, optimizer.comfort_metrics, maximize=True
        )
        
        results.append({
            'solution_id': i,
            'material_proportions': material_proportions,
            'environmental_score': -environmental_score,  # Convert back to positive
            'cost_score': -cost_score,  # Convert back to positive
            'durability_score': durability_score,
            'comfort_score': comfort_score,
            'pareto_rank': float(res.F[i][0]) if hasattr(res, 'F') and isinstance(res.F[i], (np.ndarray, list)) else float(res.F[i]) if hasattr(res, 'F') else 0
        })
    
    # Sort by Pareto rank
    results.sort(key=lambda x: x['pareto_rank'])
    
    return {
        'solutions': results,
        'best_solution': results[0] if results else None,
        'optimizer': optimizer,
        'algorithm': res
    }

def compare_with_commercial_blends(optimized_blend: Dict[str, Any], 
                                 commercial_blends: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare optimized blend with commercial blends
    
    Args:
        optimized_blend: Results from optimization
        commercial_blends: List of commercial blend data
        
    Returns:
        Comparison results
    """
    # Fix ambiguous numpy array/dict truth value
    if optimized_blend is None or len(optimized_blend) == 0 or commercial_blends is None or len(commercial_blends) == 0:
        return {}
    
    best_solution = optimized_blend['best_solution']
    comparisons = []
    
    for commercial in commercial_blends:
        comparison = {
            'commercial_name': commercial['name'],
            'commercial_blend': commercial['blend'],
            'environmental_improvement': (
                (commercial['environmental_score'] - best_solution['environmental_score']) / 
                commercial['environmental_score'] * 100
            ),
            'cost_improvement': (
                (commercial['cost_score'] - best_solution['cost_score']) / 
                commercial['cost_score'] * 100
            ),
            'durability_improvement': (
                (best_solution['durability_score'] - commercial['durability_score']) / 
                commercial['durability_score'] * 100
            ),
            'comfort_improvement': (
                (best_solution['comfort_score'] - commercial['comfort_score']) / 
                commercial['comfort_score'] * 100
            )
        }
        comparisons.append(comparison)
    
    return {
        'optimized_blend': best_solution,
        'comparisons': comparisons
    }
