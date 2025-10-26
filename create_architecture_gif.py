#!/usr/bin/env python3
"""
Create an animated GIF showing Aura's architecture and data flow
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64

def create_aura_architecture_gif():
    """Create an animated GIF showing Aura's architecture flow"""
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    colors = {
        'instagram': '#E4405F',
        'api_gateway': '#00D4AA', 
        'rabbitmq': '#FF6600',
        'worker': '#4A90E2',
        'database': '#7B68EE',
        'vector': '#FF69B4',
        'cache': '#DC143C',
        'vault': '#8B4513'
    }
    
    # Define components with positions
    components = {
        'instagram': {'pos': (1, 8), 'size': (1.5, 0.8), 'label': 'Instagram\nWebhooks'},
        'api_gateway': {'pos': (4, 8), 'size': (1.5, 0.8), 'label': 'API Gateway\n(FastAPI)'},
        'rabbitmq': {'pos': (7, 8), 'size': (1.5, 0.8), 'label': 'RabbitMQ\nMessage Broker'},
        'intelligence': {'pos': (4, 5.5), 'size': (1.5, 0.8), 'label': 'Intelligence\nWorker'},
        'ingestion': {'pos': (7, 5.5), 'size': (1.5, 0.8), 'label': 'Ingestion\nWorker'},
        'postgres': {'pos': (1, 3), 'size': (1.5, 0.8), 'label': 'PostgreSQL\n(Tenant Data)'},
        'milvus': {'pos': (4, 3), 'size': (1.5, 0.8), 'label': 'Milvus\n(Vector Search)'},
        'redis': {'pos': (7, 3), 'size': (1.5, 0.8), 'label': 'Redis\n(Cache & Locks)'},
        'vault': {'pos': (10, 3), 'size': (1.5, 0.8), 'label': 'HashiCorp\nVault'},
        'response': {'pos': (11, 8), 'size': (1.5, 0.8), 'label': 'AI Response\nBack to User'}
    }
    
    # Animation data
    frames = []
    flow_paths = [
        # Main flow
        [('instagram', 'api_gateway'), ('api_gateway', 'rabbitmq'), 
         ('rabbitmq', 'intelligence'), ('intelligence', 'response')],
        # Data flow
        [('ingestion', 'postgres'), ('ingestion', 'milvus'), 
         ('intelligence', 'milvus'), ('intelligence', 'redis')],
        # Security flow
        [('vault', 'api_gateway'), ('vault', 'intelligence'), ('vault', 'ingestion')]
    ]
    
    def animate(frame):
        ax.clear()
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Draw components
        for comp_name, comp_data in components.items():
            x, y = comp_data['pos']
            w, h = comp_data['size']
            color = colors.get(comp_name.split('_')[0], '#666666')
            
            # Create rounded rectangle
            rect = patches.FancyBboxPatch(
                (x-w/2, y-h/2), w, h,
                boxstyle="round,pad=0.1",
                facecolor=color,
                edgecolor='white',
                linewidth=2,
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add label
            ax.text(x, y, comp_data['label'], 
                   ha='center', va='center', 
                   fontsize=9, fontweight='bold', color='white')
        
        # Draw animated flow lines
        flow_idx = frame % len(flow_paths)
        current_flow = flow_paths[flow_idx]
        
        for i, (start, end) in enumerate(current_flow):
            start_pos = components[start]['pos']
            end_pos = components[end]['pos']
            
            # Animate the flow
            progress = (frame % 30) / 30.0
            if i == 0:  # First arrow in the flow
                mid_x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
                mid_y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress
                
                # Draw arrow
                ax.annotate('', xy=(mid_x, mid_y), 
                           xytext=(start_pos[0], start_pos[1]),
                           arrowprops=dict(arrowstyle='->', 
                                         color='yellow', 
                                         lw=3, alpha=0.8))
        
        # Add title
        ax.text(7, 9.5, 'Aura: AI-Powered Instagram Commerce Assistant', 
               ha='center', va='center', 
               fontsize=16, fontweight='bold', color='#333')
        
        # Add flow description
        flow_descriptions = [
            'Main Message Flow: Instagram → AI Processing → Response',
            'Data Pipeline: Product Sync → Vector Search → Context',
            'Security Flow: Vault → Secure Secrets → All Services'
        ]
        ax.text(7, 0.5, flow_descriptions[flow_idx], 
               ha='center', va='center', 
               fontsize=12, style='italic', color='#666')
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=90, interval=200, repeat=True)
    
    # Save as GIF
    anim.save('aura-architecture-flow.gif', writer='pillow', fps=5)
    plt.close()
    
    print("✅ Animated GIF created: aura-architecture-flow.gif")

if __name__ == "__main__":
    create_aura_architecture_gif()
