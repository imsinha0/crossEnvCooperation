# Test script to fix visualization updates
import time
import matplotlib.pyplot as plt
import numpy as np

# Monkey patch the show_img method to force proper updates
def create_patched_visualizer():
    from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
    from jaxmarl.viz.window import Window
    
    # Store original method
    original_show_img = Window.show_img
    
    def patched_show_img(self, img):
        """Force update the image display"""
        if self.imshow_obj is None:
            self.imshow_obj = self.ax.imshow(img, interpolation='bilinear')
            self.ax.set_xlim(0, img.shape[1])
            self.ax.set_ylim(img.shape[0], 0)
        else:
            # Force update by setting data and limits
            self.imshow_obj.set_data(img)
            self.imshow_obj.set_clim(vmin=0, vmax=255)
        
        # Force redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.1)  # Longer pause
    
    # Apply patch
    Window.show_img = patched_show_img
    
    return OvercookedVisualizer

# Usage:
# viz_class = create_patched_visualizer()
# viz = viz_class()

