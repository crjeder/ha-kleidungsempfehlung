### Requirement: Engine has a single canonical source
The system SHALL maintain engine logic in exactly one file: `custom_components/kleidungsempfehlung/engine.py`. The root `engine.py` SHALL be a re-export shim only and SHALL contain no engine logic of its own.

#### Scenario: Root engine imports resolve correctly
- **WHEN** `main.py` imports `SmartClothingEngine`, `Weather`, `SLOTS`, `get_item_id`, `is_locked`, or `make_entry` from root `engine.py`
- **THEN** all names resolve to the implementations in `custom_components/kleidungsempfehlung/engine.py`

#### Scenario: HA component is unaffected
- **WHEN** the HA sensor platform loads and imports from `.engine` via relative import
- **THEN** it imports directly from `custom_components/kleidungsempfehlung/engine.py` without any indirection through the root shim

#### Scenario: CLI produces identical output before and after
- **WHEN** `main.py` is invoked with any valid arguments
- **THEN** the output is byte-for-byte identical to the output produced before the shim was introduced
