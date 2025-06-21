import sqlite3
import json
import pandas as pd
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import logging

class FabricDatabase:
    def __init__(self, db_path: str = "data/fabric_database.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize the database with tables and sample data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create materials table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS materials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                category TEXT NOT NULL,
                description TEXT
            )
        ''')
        
        # Create metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                material_id INTEGER,
                metric_name TEXT NOT NULL,
                value REAL,
                unit TEXT,
                source_explanation TEXT,
                source_link TEXT,
                FOREIGN KEY (material_id) REFERENCES materials (id)
            )
        ''')
        
        # Create metric_categories table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metric_categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT UNIQUE NOT NULL,
                category TEXT NOT NULL,
                unit TEXT,
                optimal_direction TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
        
        # Insert sample data if database is empty
        self._insert_sample_data()
    
    def _insert_sample_data(self):
        """Insert sample fabric materials and metrics data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if data already exists
        cursor.execute("SELECT COUNT(*) FROM materials")
        if cursor.fetchone()[0] > 0:
            conn.close()
            return
        
        # Insert metric categories
        metric_categories = [
            # Environmental metrics
            ("water_consumption", "Environmental", "L/kg fiber", "minimize"),
            ("ghg_emissions", "Environmental", "kg CO2e/kg fiber", "minimize"),
            ("land_use", "Environmental", "m²/kg fiber", "minimize"),
            ("pesticide_usage", "Environmental", "kg/hectare", "minimize"),
            ("biodegradation_time", "Environmental", "months", "minimize"),
            ("energy_consumption", "Environmental", "kWh/kg", "minimize"),
            
            # Durability metrics
            ("tensile_strength", "Durability", "MPa", "maximize"),
            ("elongation_at_break", "Durability", "%", "maximize"),
            ("youngs_modulus", "Durability", "GPa", "maximize"),
            ("abrasion_cycles", "Durability", "cycles", "maximize"),
            ("burst_strength", "Durability", "kPa", "maximize"),
            ("uv_resistance", "Durability", "% strength retained", "maximize"),
            
            # Comfort metrics
            ("moisture_regain", "Comfort", "%", "maximize"),
            ("air_permeability", "Comfort", "L/s/m²", "maximize"),
            ("thermal_conductivity", "Comfort", "W/(m·K)", "maximize"),
            ("wicking_rate", "Comfort", "mm/second", "maximize"),
            ("static_resistance", "Comfort", "ohms", "maximize"),
            ("uv_protection_factor", "Comfort", "UPF", "maximize"),
            
            # Cost metrics
            ("raw_material_cost", "Cost", "$/kg", "minimize"),
            ("processing_cost", "Cost", "$/kg", "minimize"),
            ("dyeing_cost", "Cost", "$/kg", "minimize"),
            ("waste_percentage", "Cost", "%", "minimize"),
            ("energy_cost", "Cost", "$/kg", "minimize"),
            ("total_manufacturing_cost", "Cost", "$/kg", "minimize")
        ]
        
        cursor.executemany(
            "INSERT OR IGNORE INTO metric_categories (metric_name, category, unit, optimal_direction) VALUES (?, ?, ?, ?)",
            metric_categories
        )
        
        # Insert sample materials
        materials = [
            ("Organic Cotton", "Plant-Based Fibers", "Natural fiber from organic cotton plants"),
            ("Recycled Polyester (rPET)", "Synthetic Fibers", "Recycled polyester from plastic bottles"),
            ("Lyocell (Tencel)", "Semi-Synthetic Fibers", "Cellulosic fiber from wood pulp"),
            ("Hemp", "Plant-Based Fibers", "Natural fiber from hemp plants"),
            ("Bamboo", "Plant-Based Fibers", "Natural fiber from bamboo plants"),
            ("Wool", "Animal-Based Fibers", "Natural fiber from sheep"),
            ("Silk", "Animal-Based Fibers", "Natural protein fiber from silkworms"),
            ("Nylon", "Synthetic Fibers", "Synthetic polyamide fiber"),
            ("Coir (Coconut Fiber)", "Plant-Based Fibers", "Natural fiber from coconut husks"),
            ("Abaca (Manila Hemp)", "Plant-Based Fibers", "Natural fiber from banana family"),
            ("Glass Fiber", "Specialty Fibers", "Synthetic fiber made from glass"),
            ("Cashmere", "Animal-Based Fibers", "Luxury fiber from cashmere goats"),
            ("Spandex", "Synthetic Fibers", "Elastic synthetic fiber"),
            ("Linen", "Plant-Based Fibers", "Natural fiber from flax plants"),
            ("Polyester", "Synthetic Fibers", "Synthetic polymer fiber")
        ]
        
        cursor.executemany(
            "INSERT OR IGNORE INTO materials (name, category, description) VALUES (?, ?, ?)",
            materials
        )
        
        conn.commit()
        
        # Insert sample metrics data (expanded values for better optimization)
        self._insert_sample_metrics(cursor)
        
        conn.commit()
        conn.close()
    
    def _insert_sample_metrics(self, cursor):
        """Insert sample metrics data for materials"""
        # Get material IDs
        cursor.execute("SELECT id, name FROM materials")
        materials = cursor.fetchall()
        
        # Sample metrics data (expanded values for better optimization)
        sample_data = {
            "Organic Cotton": {
                "water_consumption": 2500, "ghg_emissions": 3.3, "land_use": 3.5,
                "pesticide_usage": 0.1, "biodegradation_time": 6, "energy_consumption": 55,
                "tensile_strength": 400, "elongation_at_break": 7, "youngs_modulus": 8.5,
                "abrasion_cycles": 15000, "burst_strength": 350, "uv_resistance": 85,
                "moisture_regain": 8.5, "air_permeability": 120, "thermal_conductivity": 0.26,
                "wicking_rate": 0.8, "static_resistance": 1e12, "uv_protection_factor": 15,
                "raw_material_cost": 2.5, "processing_cost": 1.2, "dyeing_cost": 0.8,
                "waste_percentage": 5, "energy_cost": 0.3, "total_manufacturing_cost": 4.8
            },
            "Recycled Polyester (rPET)": {
                "water_consumption": 100, "ghg_emissions": 2.1, "land_use": 0.1,
                "pesticide_usage": 0, "biodegradation_time": 200, "energy_consumption": 45,
                "tensile_strength": 600, "elongation_at_break": 25, "youngs_modulus": 12.0,
                "abrasion_cycles": 25000, "burst_strength": 450, "uv_resistance": 90,
                "moisture_regain": 0.4, "air_permeability": 80, "thermal_conductivity": 0.14,
                "wicking_rate": 1.2, "static_resistance": 1e10, "uv_protection_factor": 25,
                "raw_material_cost": 1.8, "processing_cost": 0.9, "dyeing_cost": 0.6,
                "waste_percentage": 3, "energy_cost": 0.2, "total_manufacturing_cost": 3.5
            },
            "Lyocell (Tencel)": {
                "water_consumption": 500, "ghg_emissions": 2.8, "land_use": 1.2,
                "pesticide_usage": 0.05, "biodegradation_time": 8, "energy_consumption": 40,
                "tensile_strength": 450, "elongation_at_break": 14, "youngs_modulus": 11.0,
                "abrasion_cycles": 18000, "burst_strength": 400, "uv_resistance": 88,
                "moisture_regain": 11.5, "air_permeability": 150, "thermal_conductivity": 0.28,
                "wicking_rate": 1.5, "static_resistance": 1e11, "uv_protection_factor": 20,
                "raw_material_cost": 3.2, "processing_cost": 1.5, "dyeing_cost": 1.0,
                "waste_percentage": 4, "energy_cost": 0.4, "total_manufacturing_cost": 6.1
            },
            "Hemp": {
                "water_consumption": 2000, "ghg_emissions": 2.2, "land_use": 2.8,
                "pesticide_usage": 0.2, "biodegradation_time": 4, "energy_consumption": 35,
                "tensile_strength": 550, "elongation_at_break": 3, "youngs_modulus": 30.0,
                "abrasion_cycles": 20000, "burst_strength": 500, "uv_resistance": 92,
                "moisture_regain": 8.0, "air_permeability": 180, "thermal_conductivity": 0.30,
                "wicking_rate": 1.0, "static_resistance": 1e12, "uv_protection_factor": 18,
                "raw_material_cost": 2.8, "processing_cost": 1.3, "dyeing_cost": 0.9,
                "waste_percentage": 6, "energy_cost": 0.3, "total_manufacturing_cost": 5.3
            },
            "Bamboo": {
                "water_consumption": 1800, "ghg_emissions": 2.5, "land_use": 2.0,
                "pesticide_usage": 0.1, "biodegradation_time": 5, "energy_consumption": 30,
                "tensile_strength": 350, "elongation_at_break": 15, "youngs_modulus": 7.0,
                "abrasion_cycles": 12000, "burst_strength": 300, "uv_resistance": 80,
                "moisture_regain": 12.0, "air_permeability": 200, "thermal_conductivity": 0.32,
                "wicking_rate": 1.8, "static_resistance": 1e11, "uv_protection_factor": 22,
                "raw_material_cost": 2.2, "processing_cost": 1.0, "dyeing_cost": 0.7,
                "waste_percentage": 4, "energy_cost": 0.2, "total_manufacturing_cost": 4.1
            },
            "Wool": {
                "water_consumption": 3000, "ghg_emissions": 13.6, "land_use": 4.5,
                "pesticide_usage": 0.3, "biodegradation_time": 2, "energy_consumption": 60,
                "tensile_strength": 300, "elongation_at_break": 35, "youngs_modulus": 2.5,
                "abrasion_cycles": 10000, "burst_strength": 250, "uv_resistance": 75,
                "moisture_regain": 16.0, "air_permeability": 90, "thermal_conductivity": 0.35,
                "wicking_rate": 0.5, "static_resistance": 1e13, "uv_protection_factor": 12,
                "raw_material_cost": 8.5, "processing_cost": 2.5, "dyeing_cost": 1.5,
                "waste_percentage": 8, "energy_cost": 0.8, "total_manufacturing_cost": 13.3
            },
            "Silk": {
                "water_consumption": 4000, "ghg_emissions": 20.0, "land_use": 5.0,
                "pesticide_usage": 0.5, "biodegradation_time": 1, "energy_consumption": 80,
                "tensile_strength": 500, "elongation_at_break": 20, "youngs_modulus": 10.0,
                "abrasion_cycles": 8000, "burst_strength": 200, "uv_resistance": 70,
                "moisture_regain": 11.0, "air_permeability": 60, "thermal_conductivity": 0.15,
                "wicking_rate": 0.3, "static_resistance": 1e14, "uv_protection_factor": 10,
                "raw_material_cost": 25.0, "processing_cost": 5.0, "dyeing_cost": 3.0,
                "waste_percentage": 12, "energy_cost": 1.2, "total_manufacturing_cost": 34.2
            },
            "Nylon": {
                "water_consumption": 150, "ghg_emissions": 5.4, "land_use": 0.1,
                "pesticide_usage": 0, "biodegradation_time": 150, "energy_consumption": 70,
                "tensile_strength": 700, "elongation_at_break": 30, "youngs_modulus": 3.0,
                "abrasion_cycles": 30000, "burst_strength": 600, "uv_resistance": 85,
                "moisture_regain": 4.5, "air_permeability": 70, "thermal_conductivity": 0.25,
                "wicking_rate": 0.9, "static_resistance": 1e9, "uv_protection_factor": 30,
                "raw_material_cost": 2.1, "processing_cost": 1.1, "dyeing_cost": 0.8,
                "waste_percentage": 2, "energy_cost": 0.3, "total_manufacturing_cost": 4.3
            },
            "Coir (Coconut Fiber)": {
                "water_consumption": 800, "ghg_emissions": 1.5, "land_use": 1.5,
                "pesticide_usage": 0.05, "biodegradation_time": 3, "energy_consumption": 25,
                "tensile_strength": 200, "elongation_at_break": 5, "youngs_modulus": 6.0,
                "abrasion_cycles": 5000, "burst_strength": 150, "uv_resistance": 95,
                "moisture_regain": 10.5, "air_permeability": 250, "thermal_conductivity": 0.40,
                "wicking_rate": 2.0, "static_resistance": 1e12, "uv_protection_factor": 25,
                "raw_material_cost": 1.2, "processing_cost": 0.8, "dyeing_cost": 0.5,
                "waste_percentage": 3, "energy_cost": 0.1, "total_manufacturing_cost": 2.6
            },
            "Abaca (Manila Hemp)": {
                "water_consumption": 1200, "ghg_emissions": 1.8, "land_use": 2.2,
                "pesticide_usage": 0.1, "biodegradation_time": 4, "energy_consumption": 28,
                "tensile_strength": 400, "elongation_at_break": 4, "youngs_modulus": 25.0,
                "abrasion_cycles": 15000, "burst_strength": 350, "uv_resistance": 90,
                "moisture_regain": 9.0, "air_permeability": 160, "thermal_conductivity": 0.29,
                "wicking_rate": 1.2, "static_resistance": 1e11, "uv_protection_factor": 20,
                "raw_material_cost": 1.8, "processing_cost": 1.0, "dyeing_cost": 0.6,
                "waste_percentage": 5, "energy_cost": 0.2, "total_manufacturing_cost": 3.6
            },
            "Glass Fiber": {
                "water_consumption": 50, "ghg_emissions": 1.2, "land_use": 0.05,
                "pesticide_usage": 0, "biodegradation_time": 1000, "energy_consumption": 100,
                "tensile_strength": 2000, "elongation_at_break": 2, "youngs_modulus": 70.0,
                "abrasion_cycles": 50000, "burst_strength": 1000, "uv_resistance": 100,
                "moisture_regain": 0.1, "air_permeability": 30, "thermal_conductivity": 0.05,
                "wicking_rate": 0.1, "static_resistance": 1e8, "uv_protection_factor": 50,
                "raw_material_cost": 4.5, "processing_cost": 2.0, "dyeing_cost": 1.2,
                "waste_percentage": 1, "energy_cost": 0.5, "total_manufacturing_cost": 8.2
            },
            "Cashmere": {
                "water_consumption": 5000, "ghg_emissions": 25.0, "land_use": 6.0,
                "pesticide_usage": 0.4, "biodegradation_time": 2, "energy_consumption": 90,
                "tensile_strength": 250, "elongation_at_break": 40, "youngs_modulus": 2.0,
                "abrasion_cycles": 6000, "burst_strength": 180, "uv_resistance": 65,
                "moisture_regain": 15.0, "air_permeability": 80, "thermal_conductivity": 0.38,
                "wicking_rate": 0.4, "static_resistance": 1e14, "uv_protection_factor": 8,
                "raw_material_cost": 45.0, "processing_cost": 8.0, "dyeing_cost": 4.0,
                "waste_percentage": 15, "energy_cost": 1.5, "total_manufacturing_cost": 58.5
            },
            "Spandex": {
                "water_consumption": 200, "ghg_emissions": 4.2, "land_use": 0.1,
                "pesticide_usage": 0, "biodegradation_time": 300, "energy_consumption": 85,
                "tensile_strength": 800, "elongation_at_break": 500, "youngs_modulus": 0.1,
                "abrasion_cycles": 8000, "burst_strength": 100, "uv_resistance": 60,
                "moisture_regain": 1.2, "air_permeability": 40, "thermal_conductivity": 0.20,
                "wicking_rate": 0.2, "static_resistance": 1e7, "uv_protection_factor": 5,
                "raw_material_cost": 12.0, "processing_cost": 3.0, "dyeing_cost": 2.0,
                "waste_percentage": 4, "energy_cost": 0.6, "total_manufacturing_cost": 17.6
            },
            "Linen": {
                "water_consumption": 2200, "ghg_emissions": 2.8, "land_use": 3.2,
                "pesticide_usage": 0.15, "biodegradation_time": 5, "energy_consumption": 38,
                "tensile_strength": 500, "elongation_at_break": 2, "youngs_modulus": 28.0,
                "abrasion_cycles": 18000, "burst_strength": 400, "uv_resistance": 88,
                "moisture_regain": 12.0, "air_permeability": 170, "thermal_conductivity": 0.31,
                "wicking_rate": 1.6, "static_resistance": 1e11, "uv_protection_factor": 23,
                "raw_material_cost": 3.5, "processing_cost": 1.4, "dyeing_cost": 1.1,
                "waste_percentage": 6, "energy_cost": 0.3, "total_manufacturing_cost": 6.3
            },
            "Polyester": {
                "water_consumption": 120, "ghg_emissions": 5.2, "land_use": 0.1,
                "pesticide_usage": 0, "biodegradation_time": 180, "energy_consumption": 65,
                "tensile_strength": 650, "elongation_at_break": 20, "youngs_modulus": 11.5,
                "abrasion_cycles": 28000, "burst_strength": 500, "uv_resistance": 88,
                "moisture_regain": 0.4, "air_permeability": 75, "thermal_conductivity": 0.16,
                "wicking_rate": 1.0, "static_resistance": 1e10, "uv_protection_factor": 28,
                "raw_material_cost": 1.5, "processing_cost": 0.8, "dyeing_cost": 0.5,
                "waste_percentage": 2, "energy_cost": 0.2, "total_manufacturing_cost": 3.0
            }
        }
        
        for material_id, material_name in materials:
            if material_name in sample_data:
                for metric_name, value in sample_data[material_name].items():
                    cursor.execute('''
                        INSERT INTO metrics (material_id, metric_name, value, source_explanation)
                        VALUES (?, ?, ?, ?)
                    ''', (material_id, metric_name, value, f"Sample data for {material_name}"))
    
    def get_all_materials(self) -> List[Dict[str, Any]]:
        """Get all materials with their metrics"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT m.id, m.name, m.category, m.description,
                   mc.metric_name, mc.category as metric_category, mc.unit, mc.optimal_direction,
                   met.value, met.source_explanation, met.source_link
            FROM materials m
            LEFT JOIN metrics met ON m.id = met.material_id
            LEFT JOIN metric_categories mc ON met.metric_name = mc.metric_name
            ORDER BY m.name, mc.metric_name
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Transform to dictionary format
        materials = {}
        for _, row in df.iterrows():
            material_name = row['name']
            if material_name not in materials:
                materials[material_name] = {
                    'id': row['id'],
                    'name': material_name,
                    'category': row['category'],
                    'description': row['description'],
                    'metrics': {}
                }
            
            if pd.notna(row['metric_name']):
                materials[material_name]['metrics'][row['metric_name']] = {
                    'value': row['value'],
                    'unit': row['unit'],
                    'category': row['metric_category'],
                    'optimal_direction': row['optimal_direction'],
                    'source_explanation': row['source_explanation'],
                    'source_link': row['source_link']
                }
        
        return list(materials.values())
    
    def get_materials_dataframe(self) -> pd.DataFrame:
        """Get materials data as a pandas DataFrame for optimization"""
        materials = self.get_all_materials()
        
        # Create DataFrame
        data = []
        for material in materials:
            row = {'material': material['name'], 'category': material['category']}
            for metric_name, metric_data in material['metrics'].items():
                row[metric_name] = metric_data['value']
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Handle missing values with KNN imputation
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            imputer = KNNImputer(n_neighbors=3)
            df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
        
        return df
    
    def search_materials(self, query: str) -> List[Dict[str, Any]]:
        """Search materials by name or category"""
        conn = sqlite3.connect(self.db_path)
        
        query_sql = '''
            SELECT DISTINCT m.id, m.name, m.category, m.description
            FROM materials m
            WHERE m.name LIKE ? OR m.category LIKE ? OR m.description LIKE ?
        '''
        
        search_term = f"%{query}%"
        cursor = conn.cursor()
        cursor.execute(query_sql, (search_term, search_term, search_term))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'name': row[1],
                'category': row[2],
                'description': row[3]
            })
        
        conn.close()
        return results
    
    def get_metric_categories(self) -> List[Dict[str, Any]]:
        """Get all metric categories"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT metric_name, category, unit, optimal_direction FROM metric_categories ORDER BY category, metric_name"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df.to_dict('records')
    
    def add_material(self, name: str, category: str, description: str = "") -> int:
        """Add a new material to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO materials (name, category, description) VALUES (?, ?, ?)",
            (name, category, description)
        )
        
        material_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return material_id
    
    def add_metric(self, material_id: int, metric_name: str, value: float, 
                   unit: str = "", source_explanation: str = "", source_link: str = ""):
        """Add a metric value for a material"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO metrics 
            (material_id, metric_name, value, unit, source_explanation, source_link)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (material_id, metric_name, value, unit, source_explanation, source_link))
        
        conn.commit()
        conn.close()
