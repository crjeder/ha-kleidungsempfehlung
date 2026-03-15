## 1. Replace Root Engine with Shim

- [x] 1.1 Replace `engine.py` (root) content with the 3-line re-export shim: `from custom_components.kleidungsempfehlung.engine import *`

## 2. Verify

- [x] 2.1 Run `python main.py --temp 10 --wind 2 --met 1.2` from the repo root and confirm it produces output without errors
- [x] 2.2 Run HA integration tests (`./ha-test/restart.sh` + `HA_TOKEN=$(cat .ha_token) ./ha-test/validate.sh`) and confirm exit 0

## 3. Commit

- [ ] 3.1 Commit with message: `Deduplicate engine: root engine.py is now a re-export shim`
- [x] 3.2 Update `claude-progress.txt`, `CHANGELOG.md`, and `TODO.md`
