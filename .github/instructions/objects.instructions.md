Add object detection and classification functionality to the bot.

Purpose:
- Detect items on the ground in the game screen via image analysis
- Move the mouse cursor over an item to read its tooltip
- Extract basic item state: identified or not, name, type, quality, key stats
- Classify each item into a value tier (S, A, B, C, D) based on general rules
- Provide an API that integrates with vision/state modules

Definitions:
- Items include drops on the ground (weapons, armor, charms, jewels, etc.)
- “Tooltip” means the in-game popup that shows stats when the mouse hovers on an item
- Quality categories in Diablo 2:
  - Unique (gold name)
  - Set (green name)
  - Rare (yellow name)
  - Magic (blue)
  - Normal (white):contentReference[oaicite:1]{index=1}

Detection:
- Use computer vision (OpenCV) to locate item name text & bounding boxes
- Distinguish color coded categories (unique/set/rare/magic/normal) from screen pixels
- Track position (x, y) of each item drop

Tooltip capture:
- Create an ItemInspector class that:
  - receives an item position
  - moves mouse to that position
  - waits a short delay
  - captures a tooltip screenshot
  - applies OCR (tesseract or similar) to extract text
  - parses these fields:
    - name
    - type (weapon, armor, charm, jewel, runeword)
    - quality (unique/set/rare/magic/normal)
    - key stats (life, damage, resistances, +skills, faster cast rate, etc.)

Classification into tiers:
- Build a rule based tieringscheme using common community heuristics:
  - **S Tier**:
    - top unique items used widely (ex: items like Harlequin Crest, Mara’s Kaleidoscope, Stone of Jordan, Hellfire Torch, Annihilus):contentReference[oaicite:2]{index=2}
    - high value runewords (e.g., Enigma, Chains of Honor)
    - items with multiple best-in-slot mods
  - **A Tier**:
    - very useful uniques or sets
    - runewords with strong overall bonuses (e.g., +skills, resistances)
    - rare items with high rolls in multiple strong stats
  - **B Tier**:
    - solid items with good stats but not best-in-slot
  - **C Tier**:
    - functional items, moderate stats (resists, moderate damage)
  - **D Tier**:
    - low value drops, no useful stats or inferior to other items

Integration:
- ItemRecognizer must add detected items to the abstract state
- The decision engine can then:
  - optionally pick items up
  - log item classification and metadata
- Provide tests and debug tools showing item boxes and parsed text on screenshots

Deliverables:
- Vision module code for item detection (item bounding boxes & color coding)
- ItemInspector class (handles moving mouse to item, OCR, parsing stats)
- Classification module with rule set
- Example integration in main.py with logs of classified items
- Overlay drawings showing item box & classification label in debug mode
- Clear docstrings for all methods


The item classification module must support an external item database JSON
so that rules for S/A/B/C/D can be updated without changing code.
