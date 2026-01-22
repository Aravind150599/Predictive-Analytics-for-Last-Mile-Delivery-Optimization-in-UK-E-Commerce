import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Define the four key challenge areas
challenges = {
    'Data Infrastructure': (2, 7),
    'Model Performance': (8, 7),
    'Organizational Culture': (2, 3),
    'Strategic Alignment': (8, 3)
}

# Define colors for each challenge
colors = {
    'Data Infrastructure': '#3498db',
    'Model Performance': '#e74c3c',
    'Organizational Culture': '#2ecc71',
    'Strategic Alignment': '#f39c12'
}

# Draw boxes for each challenge
boxes = {}
for challenge, (x, y) in challenges.items():
    box = FancyBboxPatch((x-0.8, y-0.5), 1.6, 1, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='black', 
                          facecolor=colors[challenge],
                          alpha=0.7,
                          linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, challenge, ha='center', va='center', 
            fontsize=11, fontweight='bold', wrap=True)
    boxes[challenge] = (x, y)

# Draw central connecting element
center_x, center_y = 5, 5
center_circle = plt.Circle((center_x, center_y), 0.8, 
                           color='#95a5a6', alpha=0.8, 
                           edgecolor='black', linewidth=2)
ax.add_patch(center_circle)
ax.text(center_x, center_y, 'AI Implementation\nChallenges', 
        ha='center', va='center', fontsize=10, fontweight='bold')

# Draw arrows connecting challenges to center and between challenges
arrow_style = 'simple,head_length=10,head_width=10,tail_width=3'

# Connect each challenge to center
for challenge, (x, y) in challenges.items():
    # Determine arrow direction
    dx = center_x - x
    dy = center_y - y
    distance = np.sqrt(dx**2 + dy**2)
    
    # Normalize and scale
    dx = dx / distance * (distance - 1.2)
    dy = dy / distance * (distance - 1.2)
    
    arrow = FancyArrowPatch((x, y), (x + dx, y + dy),
                           arrowstyle=arrow_style,
                           color=colors[challenge],
                           alpha=0.6,
                           linewidth=2)
    ax.add_patch(arrow)

# Draw interconnecting arrows between adjacent challenges
connections = [
    ('Data Infrastructure', 'Model Performance'),
    ('Data Infrastructure', 'Organizational Culture'),
    ('Model Performance', 'Strategic Alignment'),
    ('Organizational Culture', 'Strategic Alignment')
]

for start, end in connections:
    x1, y1 = challenges[start]
    x2, y2 = challenges[end]
    
    # Calculate offset for arrow
    dx = x2 - x1
    dy = y2 - y1
    distance = np.sqrt(dx**2 + dy**2)
    
    # Start and end points with offset
    offset = 0.9
    start_x = x1 + (dx / distance) * offset
    start_y = y1 + (dy / distance) * offset
    end_x = x2 - (dx / distance) * offset
    end_y = y2 - (dy / distance) * offset
    
    arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                           arrowstyle='<->',
                           color='gray',
                           alpha=0.4,
                           linewidth=1.5,
                           linestyle='--')
    ax.add_patch(arrow)

# Add title
plt.title('Conceptual Framework: Interconnected AI Implementation Challenges',
         fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()
