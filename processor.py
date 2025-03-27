    def __init__(self, 
                 config=None,
                 custom_prompts=None,
                 window_mode="vertical",  # "whole", "vertical", "horizontal", "quadrant"
                 window_regions=None,     # For custom window mode
                 selected_windows=None,   # List of windows to process, e.g. ["top", "bottom_right"]
                 force_cpu=False):
        """Initialize the QwenPayslipProcessor with configuration
        
        Args:
            config (dict): Custom configuration (will be merged with defaults)
            custom_prompts (dict): Custom prompts for different window positions
            window_mode (str): How to split images - "whole", "vertical", "horizontal", "quadrant"
            window_regions (list): List of regions for custom window mode. Each region is a dict with:
                                  {'name': 'region_name', 'coords': (x1, y1, x2, y2)}
                                  where coords are percentages (0.0-1.0) of image dimensions
                                  Example: [
                                      {'name': 'header', 'coords': (0.0, 0.0, 1.0, 0.2)},
                                      {'name': 'body', 'coords': (0.0, 0.2, 1.0, 0.8)},
                                      {'name': 'footer', 'coords': (0.0, 0.8, 1.0, 1.0)}
                                  ]
            selected_windows (list or str): Window positions to process (default: process all windows)
                                     Can be a list ["top", "bottom"] or a single string "top"
                                     Valid options depend on window_mode:
                                     - "vertical": ["top", "bottom"]
                                     - "horizontal": ["left", "right"]
                                     - "quadrant": ["top_left", "top_right", "bottom_left", "bottom_right"]
                                     - "custom": use the 'name' values from window_regions
            force_cpu (bool): Whether to force CPU usage even if GPU is available
        """ 