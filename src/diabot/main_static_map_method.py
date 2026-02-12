    def _try_load_static_map(self, zone_name: str):
        """Try to load static map and annotations for a zone."""
        import json
        
        # Find static map
        static_map_path = load_zone_static_map(zone_name)
        
        if static_map_path is None:
            if self.debug:
                print(f"[BOT] No static map found for {zone_name}")
            return
        
        # Load annotations
        annotations_path = static_map_path.parent / f"{static_map_path.stem}_annotations.json"
        
        if not annotations_path.exists():
            if self.debug:
                print(f"[BOT] No annotations found for {zone_name}")
            return
        
        try:
            with open(annotations_path, 'r') as f:
                self.static_map_annotations = json.load(f)
            
            # Initialize localizer
            self.static_localizer = StaticMapLocalizer(static_map_path, debug=self.debug)
            
            if self.debug:
                pois = self.static_map_annotations.get('pois', [])
                print(f"[BOT] ✓ Loaded static map for {zone_name} with {len(pois)} POIs")
                
                # List POIs
                for poi in pois:
                    print(f"      - {poi['name']} ({poi['type']}) at {poi['position']}")
            
            # Set default target (first exit)
            exits = [p for p in self.static_map_annotations.get('pois', []) if p['type'] == 'exit']
            if exits:
                self.current_target_poi = exits[0]
                if self.debug:
                    print(f"[BOT] Default target: {self.current_target_poi['name']}")
        
        except Exception as e:
            print(f"[BOT] ⚠️  Error loading static map: {e}")
            self.static_localizer = None
            self.static_map_annotations = None
