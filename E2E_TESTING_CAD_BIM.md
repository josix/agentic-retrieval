# E2E 測試流程：CAD／BIM 索引與檢索

驗證 `.dxf`、`.dwg`、`.ifc`、`.ids` 能被索引、能用描述查到，而且回來的說明文字是解析器真的讀到的內容。格式比照 `E2E_TESTING.md`：每個情境都有可直接複製的指令與預期結果。文中的預期輸出都是 2026-09-21 在本 worktree實際跑出來的。

## 前置條件

- 已安裝 `uv`
- 在本 worktree 根目錄執行：`cd ~/personal/profcoo/retrieval-skill/.claude/worktrees/cad-bim`
- 情境 5、6 需要 `dwg2dxf`（`brew install libredwg`，本機已裝）與網路（下載一個公開 DWG）

先在同一個 shell 定義這三樣，後面所有情境都會用到：

```bash
T=$(mktemp -d); mkdir -p "$T/bim" "$T/drawings" "$T/legacy"

# 裝了 cad + bim extras 的環境
function cadrun { uv run --directory engine --extra cad --extra bim "$@" }
# 完全乾淨、沒有任何 extras 的環境
function barerun { uv run --isolated --directory engine "$@" 2>&1 | grep -v "^Installed" }
```

> 用 function 而不是 `EXTRAS="--extra cad --extra bim"` 變數：zsh 不會把未加引號的變數拆成多個參數，`uv run $EXTRAS` 會直接報 `unexpected argument`。

| 情境 | 驗證什麼 | 需要 |
|---|---|---|
| 0 | 基礎 gate | — |
| 1 | IDS：零依賴就能索引與查詢 | — |
| 2 | 缺後端時寫 stub，不會壞 | — |
| 3 | 裝了後端後 stub 自動修復；用描述查到正確圖面 | cad + bim |
| 4 | 說明文字的內容正確，且不猜 | cad + bim |
| 5 | 原生 DWG，並與同圖的 DXF 對照 | + dwg2dxf、網路 |
| 6 | 沒有 DWG 轉檔器時的行為 | 同上 |
| 7 | 檔案改了會自動重新解析 | cad |
| 8 | 後備路徑：手動註冊說明文字 | cad |
| 9 | 在 Claude Code 裡透過 skill 使用（手動） | plugin |

---

## 情境 0：基礎 gate

```bash
cadrun python -m unittest discover -s tests
barerun python -m unittest tests.test_cad_extractors
uv run --directory engine --extra dev ruff check .
uv run --project engine --group docs mkdocs build --strict -f mkdocs.yml
```

**預期**：
- 第一行：`Ran 429 tests` … `OK (skipped=48)`
- 第二行（乾淨環境）：一樣 `OK`，但 DXF／IFC 測試會 skip；IDS 與 wiring 測試照跑。這證明核心在沒裝 ezdxf／ifcopenshell 時仍可匯入、可運作
- ruff：`All checks passed!`；mkdocs：無 strict 錯誤

---

## 情境 1：IDS，零依賴

`.ids` 是 XML，用標準函式庫解析，不需要任何 extra。

```bash
cat > "$T/bim/doors.ids" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<ids xmlns="http://standards.buildingsmart.org/IDS" xmlns:xs="http://www.w3.org/2001/XMLSchema">
  <info><title>Door requirements</title></info>
  <specifications>
    <specification name="Door fire rating" ifcVersion="IFC4">
      <applicability minOccurs="1" maxOccurs="unbounded">
        <entity><name><simpleValue>IFCDOOR</simpleValue></name></entity>
      </applicability>
      <requirements>
        <property dataType="IFCLABEL" cardinality="required">
          <propertySet><simpleValue>Pset_DoorCommon</simpleValue></propertySet>
          <baseName><simpleValue>FireRating</simpleValue></baseName>
          <value><simpleValue>60min</simpleValue></value>
        </property>
      </requirements>
    </specification>
  </specifications>
</ids>
EOF

barerun retrieval index --root "$T" --retriever lexical
barerun retrieval query "door fire rating requirement" --root "$T" --retriever lexical --top-k 1
barerun retrieval sidecar --list --root "$T"
cat "$T/.agentic-retrieval/extracted/bim/doors.ids.md"
```

**預期**：
- `query` 第一筆：`.agentic-retrieval/extracted/bim/doors.ids.md:4-6`
- `sidecar --list`：`bim/doors.ids: ok`
- sidecar 內文用文字寫出規格：`Applies to elements matching:` → `entity: name \`IFCDOOR\``；`Each must satisfy:` → `property: propertySet \`Pset_DoorCommon\`; baseName \`FireRating\`; value \`60min\` [required] (dataType IFCLABEL)`；`Occurrence: required (at least one matching element)`

**對應關係**（之後每個情境都一樣）：命中路徑去掉前綴 `.agentic-retrieval/extracted/` 和結尾 `.md`，就是原始檔 `bim/doors.ids`。

---

## 情境 2：缺後端時寫 stub

先產生測試圖面與模型（這一步本身需要 extras，因為要用 ezdxf／ifcopenshell 寫檔）：

```bash
cat > "$T/../make_fixtures.py" <<'EOF'
import sys
from pathlib import Path
import ezdxf
import ifcopenshell.api

root = Path(sys.argv[1])

def plate(path, units=4, note="4X DIA 8 THRU"):
    doc = ezdxf.new("R2018"); doc.header["$INSUNITS"] = units
    for name in ("PROFILE", "HOLES", "NOTE"):
        doc.layers.new(name)
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (100, 0), (100, 60), (0, 60)], close=True, dxfattribs={"layer": "PROFILE"})
    for x, y in [(15, 15), (85, 15), (15, 45), (85, 45)]:
        msp.add_circle((x, y), 3, dxfattribs={"layer": "HOLES"})
    msp.add_text(note, dxfattribs={"height": 4, "insert": (4, 70), "layer": "NOTE"})
    doc.saveas(path)

plate(root / "drawings/plate_conflict.dxf")                  # 註記寫 Ø8，實際孔是 Ø6
plate(root / "drawings/plate_ok.dxf", note="4X DIA 6 THRU")  # 註記與幾何一致
plate(root / "drawings/plate_no_units.dxf", units=0)         # 單位不明

run = ifcopenshell.api.run
model = run("project.create_file", version="IFC4")
run("root.create_entity", model, ifc_class="IfcProject", name="Training building")
run("unit.assign_unit", model)
storey = run("root.create_entity", model, ifc_class="IfcBuildingStorey", name="L1")
for name, rating in [("D01", "60min"), ("D02", None), ("D03", "30min")]:
    door = run("root.create_entity", model, ifc_class="IfcDoor", name=name)
    run("spatial.assign_container", model, products=[door], relating_structure=storey)
    if rating:
        pset = run("pset.add_pset", model, product=door, name="Pset_DoorCommon")
        run("pset.edit_pset", model, pset=pset, properties={"FireRating": rating})
model.write(str(root / "bim/doors.ifc"))
print("fixtures written to", root)
EOF
cadrun python "$T/../make_fixtures.py" "$T"
```

然後用**沒有 extras** 的環境索引：

```bash
barerun retrieval index --root "$T" --retriever lexical
barerun retrieval sidecar --list --root "$T"
cat "$T/.agentic-retrieval/extracted/drawings/plate_ok.dxf.md"
```

**預期**：
- `index` 成功（exit 0），stderr 各出現一次：
  `warning: ifcopenshell is not available - .ifc sidecars are written as stub placeholders …`
  `warning: ezdxf is not available - .dxf sidecars are written as stub placeholders …`
- `sidecar --list`：`bim/doors.ids: ok`，其餘四個檔案都是 `stub (backend-missing) -> needs transcript`
- stub 內文是安裝指引（`Install the 'cad' extra (\`uv pip install -e '.[cad]'\`) and reindex.`），不是空白，也沒有假造的內容

---

## 情境 3：stub 自動修復，用描述查圖

直接換成有 extras 的環境查詢，不需要手動重建或清快取：

```bash
cadrun retrieval query "plate where the hole note disagrees with the drilled holes" \
  --root "$T" --retriever lexical --top-k 3
cadrun retrieval sidecar --list --root "$T"
cadrun retrieval query "drawing with unknown units" --root "$T" --retriever lexical --top-k 1
cadrun retrieval query "doors missing a fire rating property" --root "$T" --retriever lexical --top-k 1
```

**預期**：
- 第一個查詢的第一筆是 `…/drawings/plate_conflict.dxf.md:44-46`（三張板子都有「Hole callouts」段，但衝突那張排第一）
- `sidecar --list` 全部變成 `ok`（後端出現後，`backend-missing` 的 stub 在下一次 index／query 自動重新解析）
- 第二個查詢：`…/drawings/plate_no_units.dxf.md:28-42`
- 第三個查詢：`…/bim/doors.ifc.md:18-22`

想看每個 retriever 的共識與信心，拿掉 `--retriever lexical`（預設是 consolidated）；輸出會多出 `[score=… agree=2/2 conf=high via lexical,treesitter]` 與命中的段落標題。第一次跑會載入較多東西，比較慢。

---

## 情境 4：說明文字的內容正確，且不猜

```bash
E="$T/.agentic-retrieval/extracted"
grep -h "CONFLICT\|matches the geometry" "$E"/drawings/*.md
grep -h "^| PROFILE | 6000" "$E"/drawings/*.md
grep -h "FireRating" "$E/bim/doors.ifc.md"
grep -o "data-only[^.]*" "$E/bim/doors.ifc.md"
```

**預期**：

```text
- CONFLICT: note '4X DIA 8 THRU' calls for 4 hole(s) of diameter 8, but the geometry has 0; circles present: 4 x diameter 6. Neither value is authoritative - ask the author.
- CONFLICT: …（plate_no_units 也是同一個註記，所以有兩行）
- Note '4X DIA 6 THRU' matches the geometry: 4 circle(s) of diameter 6.
| PROFILE | 6000 | 0.006 m2 | 4 | 32 |                          ← plate_conflict、plate_ok（mm）
| PROFILE | 6000 | not computed (unknown units) | 4 | 32 |      ← plate_no_units
| Pset_DoorCommon | FireRating | 2/3 (1 missing) | 60min x1, 30min x1 |
data-only, so no quantity or clash can be derived from it
```

要人工確認的三件事：
1. **不猜單位**：`plate_no_units` 沒有任何 `m2` 數值，開頭寫 `unknown ($INSUNITS=0)`，Caveats 寫 `Do not assume millimeters.`
2. **不下結論**：衝突那行說的是「兩個值都不算數，去問作者」，不是替你選 6 或 8
3. **可追溯**：表格裡有 handle（`32`、`37`）、IFC 有 GlobalId 與 STEP id（`#12`），可以回原檔對

完整讀一份：`cat "$E/drawings/plate_conflict.dxf.md"`（約 60 行）。

---

## 情境 5：原生 DWG，與同圖的 DXF 對照

LibreDWG 測試資料裡有同一張圖的 DWG 與 DXF（GPL-3.0，這裡只下載到暫存目錄，不進 repo）：

```bash
B=https://raw.githubusercontent.com/LibreDWG/libredwg/34f02f54b9aacb5708c1d3d2070efb3e4b2d8c43/test/test-data
curl -sfL -o "$T/legacy/example_2018.dwg" "$B/example_2018.dwg"
curl -sfL -o "$T/legacy/example_2018.dxf" "$B/example_2018.dxf"

cadrun retrieval extract --root "$T" | grep legacy
sed -n 6p "$T/.agentic-retrieval/extracted/legacy/example_2018.dwg.md" | cut -c1-125
sed -n 6p "$T/.agentic-retrieval/extracted/legacy/example_2018.dxf.md" | cut -c1-125
tail -3 "$T/.agentic-retrieval/extracted/legacy/example_2018.dwg.md"
cadrun retrieval query "Esperanto text teksto simpla in a native DWG" --root "$T" --retriever lexical --top-k 1
```

**預期**：
- `extract`：兩個檔案都是 `-> …md (NNNN chars)`，不是 stub
- DWG：`DWG drawing, format AC1032 (AutoCAD R2018). Drawing units: millimeters … Model space holds 67 entities`
- DXF：`DXF drawing, … Model space holds 68 entities`
- DWG 結尾的 `## Conversion` 段：`Native DWG release AC1032 (AutoCAD 2018), converted to DXF by dwg2dxf … counts are a lower bound on the native file.`
- 查詢命中 `…/legacy/example_2018.dwg.md`（圖裡的文字是世界語，會逐字出現在 Text 表）

**67 對 68 是真的差異，不是 bug**：少掉的那一個是 `ACAD_TABLE`，`dwg2dxf` 沒有轉出來。這正是 Conversion 段那句「下限」要提醒的事。同一張圖的 2000 版則是 68 對 68 完全一致。

---

## 情境 6：沒有 DWG 轉檔器

用一個不含 Homebrew 的 `PATH` 模擬沒裝 `dwg2dxf` 的機器（`uv` 在 `~/.local/bin`；你的若不同請調整）：

```bash
env PATH="$HOME/.local/bin:/usr/bin:/bin" \
  uv run --directory engine --extra cad --extra bim retrieval extract --root "$T" --force 2>&1 | grep -i dwg
grep -o "DWG release[^.]*" "$T/.agentic-retrieval/extracted/legacy/example_2018.dwg.md"

cadrun retrieval index --root "$T" --retriever lexical > /dev/null
cadrun retrieval sidecar --list --root "$T" | grep dwg
cadrun retrieval sidecar --list --root "$T" --json | python3 -c "import sys,json; print(json.load(sys.stdin)['backends'])"
```

**預期**：
- `warning: DWG converter is not available …` 與 `legacy/example_2018.dwg: stub (backend-missing)`，其他檔案照常解析，指令 exit 0
- 即使讀不了內容，stub 仍回報 `DWG release: AC1032 (AutoCAD 2018)`（只讀檔頭 6 個 byte）
- 換回正常 `PATH` 重新索引後：`legacy/example_2018.dwg: ok`
- `backends`：`{'.dwg': True, '.dxf': True, '.ids': True, '.ifc': True, '.pdf': False}`（`.pdf` 為 False 是因為沒裝 pdf extra，與本功能無關）

---

## 情境 7：檔案改了會自動重新解析

```bash
cadrun python - "$T" <<'EOF'
import sys, ezdxf
p = sys.argv[1] + "/drawings/plate_ok.dxf"
doc = ezdxf.readfile(p)
doc.modelspace().add_text("MATERIAL AL6061 T6", dxfattribs={"layer": "NOTE", "height": 4, "insert": (4, 80)})
doc.saveas(p)
EOF
cadrun retrieval query "AL6061 material" --root "$T" --retriever lexical --top-k 1
```

**預期**：不必手動 `index`，直接查就命中 `…/drawings/plate_ok.dxf.md:20-27`（Text 表），因為內容指紋變了，`query` 會先重建。

---

## 情境 8：後備路徑，手動註冊說明文字

裝不了後端、但手上有別的工具輸出的摘要時使用。

```bash
printf 'Vendor viewer export: site plan, 3 buildings, north arrow, scale 1:500.\n' > "$T/../t.md"
cadrun retrieval sidecar --register legacy/example_2018.dwg --transcript "$T/../t.md" --root "$T"
cadrun retrieval sidecar --list --root "$T" | grep dwg
cadrun retrieval query "site plan north arrow scale 1:500" --root "$T" --retriever lexical --top-k 1
cadrun retrieval extract --root "$T" --force 2>&1 | grep -i "warning\|dwg"
```

**預期**：
- `legacy/example_2018.dwg -> .agentic-retrieval/extracted/legacy/example_2018.dwg.md`
- `--list`：`legacy/example_2018.dwg: agent-authored`；之後一般的 index／query **不會**用機器解析蓋掉它
- 查詢命中 `…/legacy/example_2018.dwg.md:1-4`
- 只有明確下 `extract --force` 才會覆寫，並先警告：`warning: overwriting 1 agent-authored sidecar(s) with machine-extracted output`

---

## 情境 9：在 Claude Code 裡透過 skill 使用（手動）

> 這一段我**沒有**實際跑過，只驗證了 skill 指示要執行的 CLI 流程（情境 1–8）。

1. 找一個放了 CAD／BIM 檔的專案（或把 `$T` 複製到固定位置），在該目錄啟動 Claude Code 並載入本 worktree 的 plugin：
   `claude --plugin-dir ~/personal/profcoo/retrieval-skill/.claude/worktrees/cad-bim`
2. 用自己的話問，例如：
   - 「哪一張圖的孔徑註記跟實際鑽孔對不上？」
   - 「找出有門沒填防火時效的模型，缺的是哪幾扇？」
   - 「`legacy/example_2018.dwg` 裡面有什麼？」

**驗收標準**（來自 `skills/cad-bim-retrieval-usage/SKILL.md`）：
- 有觸發 `cad-bim-retrieval-usage` 或 `/retrieval`，並且真的執行了 `retrieval query`
- 回答給的是**原始檔**路徑（`drawings/plate_conflict.dxf`），不是 sidecar 路徑
- 有引用證據：圖層名、handle、GlobalId
- 有講出限制：例如 DWG 是轉檔後的結果、`60min` 只是字串不代表符合法規
- 把 `CONFLICT` 或 `2/3 (1 missing)` 當成要問作者的問題，而不是替作者下結論
- 遇到 stub 時說「讀不到、需要裝什麼」，而不是編內容

---

## 收尾

```bash
rm -rf "$T" "$T/../make_fixtures.py" "$T/../t.md"
```

測試過程只寫入 `$T`；索引與 sidecar 都在 `$T/.agentic-retrieval/` 底下。

## 失敗時先看這裡

| 現象 | 原因 |
|---|---|
| `error: unexpected argument '--extra cad --extra bim'` | 在 zsh 用了未拆分的變數；改用上面的 `cadrun` function |
| 裝了 extras 但還是 `stub (backend-missing)` | `uv run` 沒帶 `--extra cad --extra bim`；每一次呼叫都要帶 |
| 只有 `.dwg` 是 stub | `dwg2dxf` 不在 `PATH`：`command -v dwg2dxf` |
| `.dwg` 是 `stub (convert-failed)` | 轉檔器跑了但產不出 DXF；該檔含轉檔器不支援的內容 |
| 改了 `cad_extractors.py` 的輸出但 sidecar 沒變 | 快取以「檔案 sha256 + extractor 版本」判定；bump `extractors.py` 裡該 Backend 的 `version`，或 `extract --force` |
| `retrieval extract` 要求 pdf extra | 目錄裡有 `.pdf`；加 `--extra pdf`，或該次用 `index --no-pdf` |
