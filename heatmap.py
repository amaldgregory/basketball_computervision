import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import pandas as pd
from scipy.ndimage import gaussian_filter
import cv2


class ShotHeatMap:
    """Generates heat maps for basketball shot analysis"""
    
    def __init__(self, court_dimensions: Tuple[float, float] = (94, 50)):
        """
        Initialize the heat map generator
        
        Args:
            court_dimensions: Court dimensions in feet (width, height)
        """
        self.court_width, self.court_height = court_dimensions
        self.shot_locations = []
        self.court_image = None
        
        # Standard basketball court measurements (NBA)
        self.court_measurements = {
            'three_point_line': 23.75,  # feet from basket
            'free_throw_line': 15,      # feet from basket
            'paint_width': 16,          # feet
            'basket_height': 10         # feet
        }
    
    def add_shot_location(self, x: float, y: float, made: bool = True, 
                         confidence: float = 1.0, shot_type: str = "jump_shot") -> None:
        """
        Add a shot location to the heat map
        
        Args:
            x: X coordinate (feet from left baseline)
            y: Y coordinate (feet from baseline)
            made: Whether the shot was made
            confidence: Confidence in the shot location
            shot_type: Type of shot (jump_shot, layup, etc.)
        """
        self.shot_locations.append({
            'x': x,
            'y': y,
            'made': made,
            'confidence': confidence,
            'shot_type': shot_type
        })
    
    def add_shot_from_pose_data(self, pose_data: Dict, made: bool = True, 
                               shot_type: str = "jump_shot") -> None:
        """
        Add shot location based on pose data analysis
        
        Args:
            pose_data: Pose detection data
            made: Whether the shot was made
            shot_type: Type of shot
        """
        # This is a simplified version - in practice, you'd need more sophisticated
        # analysis to determine shot location from pose data
        if 'keypoints' in pose_data:
            # Use wrist position as approximate shot location
            if 'right_wrist' in pose_data['keypoints']:
                wrist = pose_data['keypoints']['right_wrist']
                # Convert pixel coordinates to court coordinates
                # This is a rough approximation
                x = (wrist['x'] / 640) * self.court_width  # Assuming 640px width
                y = (wrist['y'] / 480) * self.court_height  # Assuming 480px height
                
                self.add_shot_location(x, y, made, wrist['confidence'], shot_type)
    
    def create_shot_frequency_heatmap(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a heat map showing shot frequency by location
        
        Args:
            save_path: Optional path to save the heat map
            
        Returns:
            Matplotlib figure object
        """
        if not self.shot_locations:
            raise ValueError("No shot locations available for heat map")
        
        # Convert to DataFrame
        df = pd.DataFrame(self.shot_locations)
        
        # Create figure with court background
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create 2D histogram
        x_bins = np.linspace(0, self.court_width, 50)
        y_bins = np.linspace(0, self.court_height, 30)
        
        heatmap, xedges, yedges = np.histogram2d(
            df['x'], df['y'], bins=[x_bins, y_bins]
        )
        
        # Apply Gaussian smoothing
        heatmap_smooth = gaussian_filter(heatmap, sigma=1)
        
        # Create heat map
        im = ax.imshow(heatmap_smooth.T, origin='lower', 
                      extent=[0, self.court_width, 0, self.court_height],
                      cmap='Reds', alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Shot Frequency', fontsize=12)
        
        # Add court markings
        self._add_court_markings(ax)
        
        # Add title and labels
        ax.set_title('Shot Frequency Heat Map', fontsize=16, fontweight='bold')
        ax.set_xlabel('Court Width (feet)', fontsize=12)
        ax.set_ylabel('Court Length (feet)', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Shot frequency heat map saved to: {save_path}")
        
        return fig
    
    def create_accuracy_heatmap(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a heat map showing shot accuracy by location
        
        Args:
            save_path: Optional path to save the heat map
            
        Returns:
            Matplotlib figure object
        """
        if not self.shot_locations:
            raise ValueError("No shot locations available for accuracy heat map")
        
        df = pd.DataFrame(self.shot_locations)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate accuracy by location
        x_bins = np.linspace(0, self.court_width, 25)
        y_bins = np.linspace(0, self.court_height, 15)
        
        # Group shots by location and calculate accuracy
        df['x_bin'] = pd.cut(df['x'], bins=x_bins, labels=False)
        df['y_bin'] = pd.cut(df['y'], bins=y_bins, labels=False)
        
        accuracy_data = df.groupby(['x_bin', 'y_bin']).agg({
            'made': ['count', 'sum']
        }).reset_index()
        
        accuracy_data.columns = ['x_bin', 'y_bin', 'total_shots', 'made_shots']
        accuracy_data['accuracy'] = accuracy_data['made_shots'] / accuracy_data['total_shots']
        
        # Create accuracy matrix
        accuracy_matrix = np.zeros((len(y_bins)-1, len(x_bins)-1))
        
        for _, row in accuracy_data.iterrows():
            if not np.isnan(row['x_bin']) and not np.isnan(row['y_bin']):
                x_idx = int(row['x_bin'])
                y_idx = int(row['y_bin'])
                if x_idx < accuracy_matrix.shape[1] and y_idx < accuracy_matrix.shape[0]:
                    accuracy_matrix[y_idx, x_idx] = row['accuracy']
        
        # Apply smoothing
        accuracy_smooth = gaussian_filter(accuracy_matrix, sigma=0.5)
        
        # Create heat map
        im = ax.imshow(accuracy_smooth, origin='lower',
                      extent=[0, self.court_width, 0, self.court_height],
                      cmap='RdYlGn', vmin=0, vmax=1, alpha=0.8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Shot Accuracy', fontsize=12)
        
        # Add court markings
        self._add_court_markings(ax)
        
        # Add title and labels
        ax.set_title('Shot Accuracy Heat Map', fontsize=16, fontweight='bold')
        ax.set_xlabel('Court Width (feet)', fontsize=12)
        ax.set_ylabel('Court Length (feet)', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Accuracy heat map saved to: {save_path}")
        
        return fig
    
    def create_shot_type_heatmap(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a heat map showing different shot types by location
        
        Args:
            save_path: Optional path to save the heat map
            
        Returns:
            Matplotlib figure object
        """
        if not self.shot_locations:
            raise ValueError("No shot locations available for shot type heat map")
        
        df = pd.DataFrame(self.shot_locations)
        
        # Get unique shot types
        shot_types = df['shot_type'].unique()
        
        # Create subplots for each shot type
        n_types = len(shot_types)
        fig, axes = plt.subplots(1, n_types, figsize=(6*n_types, 8))
        
        if n_types == 1:
            axes = [axes]
        
        for i, shot_type in enumerate(shot_types):
            ax = axes[i]
            
            # Filter data for this shot type
            type_data = df[df['shot_type'] == shot_type]
            
            if len(type_data) > 0:
                # Create heat map for this shot type
                x_bins = np.linspace(0, self.court_width, 30)
                y_bins = np.linspace(0, self.court_height, 20)
                
                heatmap, _, _ = np.histogram2d(
                    type_data['x'], type_data['y'], bins=[x_bins, y_bins]
                )
                
                # Apply smoothing
                heatmap_smooth = gaussian_filter(heatmap, sigma=1)
                
                # Create heat map
                im = ax.imshow(heatmap_smooth.T, origin='lower',
                              extent=[0, self.court_width, 0, self.court_height],
                              cmap='Blues', alpha=0.7)
                
                # Add court markings
                self._add_court_markings(ax)
                
                ax.set_title(f'{shot_type.replace("_", " ").title()} Shots', 
                           fontsize=14, fontweight='bold')
                ax.set_xlabel('Court Width (feet)', fontsize=10)
                ax.set_ylabel('Court Length (feet)', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Shot type heat map saved to: {save_path}")
        
        return fig
    
    def _add_court_markings(self, ax: plt.Axes) -> None:
        """
        Add basketball court markings to the plot
        
        Args:
            ax: Matplotlib axes object
        """
        # Court outline
        ax.add_patch(plt.Rectangle((0, 0), self.court_width, self.court_height, 
                                  fill=False, color='black', linewidth=2))
        
        # Half court line
        mid_court = self.court_width / 2
        ax.axvline(mid_court, color='black', linewidth=2, alpha=0.7)
        
        # Three-point lines (simplified)
        three_point_dist = self.court_measurements['three_point_line']
        ax.add_patch(plt.Circle((mid_court, three_point_dist), three_point_dist, 
                               fill=False, color='red', linewidth=2, alpha=0.7))
        ax.add_patch(plt.Circle((mid_court, self.court_height - three_point_dist), 
                               three_point_dist, fill=False, color='red', 
                               linewidth=2, alpha=0.7))
        
        # Free throw circles
        ft_dist = self.court_measurements['free_throw_line']
        ax.add_patch(plt.Circle((mid_court, ft_dist), 6, 
                               fill=False, color='black', linewidth=1, alpha=0.7))
        ax.add_patch(plt.Circle((mid_court, self.court_height - ft_dist), 6, 
                               fill=False, color='black', linewidth=1, alpha=0.7))
        
        # Paint areas
        paint_width = self.court_measurements['paint_width']
        paint_left = mid_court - paint_width / 2
        paint_right = mid_court + paint_width / 2
        
        # Top paint
        ax.add_patch(plt.Rectangle((paint_left, 0), paint_width, ft_dist, 
                                  fill=False, color='black', linewidth=1, alpha=0.7))
        
        # Bottom paint
        ax.add_patch(plt.Rectangle((paint_left, self.court_height - ft_dist), 
                                  paint_width, ft_dist, fill=False, 
                                  color='black', linewidth=1, alpha=0.7))
    
    def get_shot_statistics(self) -> Dict:
        """
        Get statistics about the shot locations
        
        Returns:
            Dictionary containing shot statistics
        """
        if not self.shot_locations:
            return {"error": "No shot locations available"}
        
        df = pd.DataFrame(self.shot_locations)
        
        # Basic statistics
        total_shots = len(df)
        made_shots = df['made'].sum()
        accuracy = made_shots / total_shots if total_shots > 0 else 0
        
        # Location statistics
        avg_x = df['x'].mean()
        avg_y = df['y'].mean()
        std_x = df['x'].std()
        std_y = df['y'].std()
        
        # Shot type distribution
        shot_type_dist = df['shot_type'].value_counts().to_dict()
        
        # Distance from basket analysis
        mid_court = self.court_width / 2
        df['distance_from_basket'] = np.sqrt((df['x'] - mid_court)**2 + (df['y'] - 0)**2)
        avg_distance = df['distance_from_basket'].mean()
        
        # Accuracy by distance
        distance_bins = [0, 10, 20, 30, 50]
        df['distance_bin'] = pd.cut(df['distance_from_basket'], bins=distance_bins, labels=False)
        accuracy_by_distance = df.groupby('distance_bin')['made'].agg(['count', 'sum']).reset_index()
        accuracy_by_distance['accuracy'] = accuracy_by_distance['sum'] / accuracy_by_distance['count']
        
        return {
            'total_shots': total_shots,
            'made_shots': made_shots,
            'overall_accuracy': accuracy,
            'average_location': (avg_x, avg_y),
            'location_std': (std_x, std_y),
            'shot_type_distribution': shot_type_dist,
            'average_distance_from_basket': avg_distance,
            'accuracy_by_distance': accuracy_by_distance.to_dict('records')
        }
    
    def clear_data(self) -> None:
        """Clear all shot location data"""
        self.shot_locations = []
    
    def load_data_from_file(self, filepath: str) -> None:
        """
        Load shot location data from a file
        
        Args:
            filepath: Path to the data file (CSV or JSON)
        """
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            self.shot_locations = df.to_dict('records')
        elif filepath.endswith('.json'):
            import json
            with open(filepath, 'r') as f:
                self.shot_locations = json.load(f)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
        
        print(f"Loaded {len(self.shot_locations)} shot locations from {filepath}")


if __name__ == "__main__":
    # Test the heat map generator
    heatmap = ShotHeatMap()
    
    # Add some example shot locations
    np.random.seed(42)
    
    # Add jump shots (mostly from mid-range)
    for _ in range(50):
        x = np.random.normal(47, 10)  # Center of court
        y = np.random.normal(15, 8)   # Mid-range
        made = np.random.random() > 0.4  # 60% accuracy
        heatmap.add_shot_location(x, y, made, shot_type="jump_shot")
    
    # Add layups (close to basket)
    for _ in range(20):
        x = np.random.normal(47, 5)   # Close to center
        y = np.random.normal(5, 3)    # Close to basket
        made = np.random.random() > 0.2  # 80% accuracy
        heatmap.add_shot_location(x, y, made, shot_type="layup")
    
    # Add three-pointers
    for _ in range(30):
        angle = np.random.uniform(0, 2*np.pi)
        distance = np.random.uniform(23, 25)  # Three-point range
        x = 47 + distance * np.cos(angle)
        y = distance * np.sin(angle)
        made = np.random.random() > 0.6  # 40% accuracy
        heatmap.add_shot_location(x, y, made, shot_type="three_pointer")
    
    # Generate heat maps
    print("Generating heat maps...")
    
    # Shot frequency heat map
    fig1 = heatmap.create_shot_frequency_heatmap()
    plt.show()
    
    # Accuracy heat map
    fig2 = heatmap.create_accuracy_heatmap()
    plt.show()
    
    # Shot type heat map
    fig3 = heatmap.create_shot_type_heatmap()
    plt.show()
    
    # Print statistics
    stats = heatmap.get_shot_statistics()
    print("\nShot Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}") 