#!/usr/bin/env python3
"""
Create a smaller, optimized animated GIF for GitHub
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np

def create_optimized_gif():
    """Create a smaller animated GIF"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Simplified components
    components = {
        'instagram': {'pos': (1, 4.5), 'size': (1.2, 0.6), 'label': 'Instagram', 'color': '#E4405F'},
        'api': {'pos': (3, 4.5), 'size': (1.2, 0.6), 'label': 'API Gateway', 'color': '#00D4AA'},
        'queue': {'pos': (5, 4.5), 'size': (1.2, 0.6), 'label': 'RabbitMQ', 'color': '#FF6600'},
        'ai': {'pos': (7, 4.5), 'size': (1.2, 0.6), 'label': 'AI Worker', 'color': '#4A90E2'},
        'response': {'pos': (9, 4.5), 'size': (1.2, 0.6), 'label': 'Response', 'color': '#32CD32'},
        'db': {'pos': (2, 2), 'size': (1.2, 0.6), 'label': 'PostgreSQL', 'color': '#7B68EE'},
        'vector': {'pos': (4, 2), 'size': (1.2, 0.6), 'label': 'Milvus', 'color': '#FF69B4'},
        'cache': {'pos': (6, 2), 'size': (1.2, 0.6), 'label': 'Redis', 'color': '#DC143C'},
        'vault': {'pos': (8, 2), 'size': (1.2, 0.6), 'label': 'Vault', 'color': '#8B4513'}
    }
    
    def animate(frame):
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis('off')
        
        # Draw components
        for comp_name, comp_data in components.items():
            x, y = comp_data['pos']
            w, h = comp_data['size']
            
            rect = patches.FancyBboxPatch(
                (x-w/2, y-h/2), w, h,
                boxstyle="round,pad=0.05",
                facecolor=comp_data['color'],
                edgecolor='white',
                linewidth=1,
                alpha=0.9
            )
            ax.add_patch(rect)
            
            ax.text(x, y, comp_data['label'], 
                   ha='center', va='center', 
                   fontsize=8, fontweight='bold', color='white')
        
        # Animated flow
        progress = (frame % 20) / 20.0
        
        # Main flow
        flow_components = ['instagram', 'api', 'queue', 'ai', 'response']
        for i in range(len(flow_components) - 1):
            start_pos = components[flow_components[i]]['pos']
            end_pos = components[flow_components[i+1]]['pos']
            
            if i == int(progress * (len(flow_components) - 1)):
                mid_x = start_pos[0] + (end_pos[0] - start_pos[0]) * (progress * (len(flow_components) - 1) - i)
                mid_y = start_pos[1] + (end_pos[1] - start_pos[1]) * (progress * (len(flow_components) - 1) - i)
                
                ax.annotate('', xy=(mid_x, mid_y), 
                           xytext=(start_pos[0], start_pos[1]),
                           arrowprops=dict(arrowstyle='->', 
                                         color='yellow', 
                                         lw=2, alpha=0.8))
        
        ax.text(5, 5.5, 'Aura Architecture Flow', 
               ha='center', va='center', 
               fontsize=12, fontweight='bold', color='#333')
    
    anim = FuncAnimation(fig, animate, frames=40, interval=150, repeat=True)
    anim.save('aura-flow-small.gif', writer='pillow', fps=6, 
              savefig_kwargs={'bbox_inches': 'tight', 'pad_inches': 0.1})
    plt.close()
    
    print("✅ Optimized GIF created: aura-flow-small.gif")

if __name__ == "__main__":
    create_optimized_gif()
