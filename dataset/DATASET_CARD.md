# VIDA grounded-model dataset

Function-calling training data for a grounded Volvo VIDA repair model. Generated
locally from a licensed VIDA install by `dataset/gen_dataset.py`; **only the produced
JSONL is shipped to the trainer — never the source DBs** (data/*.sqlite3 are gitignored).

## Format (one JSON object per line)
The canonical contract record from `grounding/contract.py` — the EXACT message shape
the model is served under (DECISIONS D1):

```
{"messages":[
  {"role":"system","content":"<grounding.contract.SYSTEM_PROMPT>"},
  {"role":"user","content":"[VEHICLE] 2004 S60 (-09) | D5244T [/VEHICLE]\n<question>"},
  {"role":"assistant","content":null,"tool_calls":[{"id":"call_0001","type":"function",
     "function":{"name":"search_docs","arguments":"{...}"}}]},
  {"role":"tool","tool_call_id":"call_0001","content":"{...real VIDA row...}"},
  {"role":"assistant","content":"30 Nm (VCC-413372-2), for the D5244T."}
],"metadata":{"example_type":"...","answer_type":"...","vehicle":{...},
              "split_key":"<model>|<topic>","source_ids":["VCC-413372-2"]}}
```
Every record passes `grounding.contract.validate_messages`. Citations are the REAL
`vida_doc_ref` (VCC-…) or `doc_id` — never invented (D7).

## Example types (target mix from `gc.EXAMPLE_TYPES`)
| type | ~share | what it teaches | source |
|---|---|---|---|
| grounded | 46% | call a tool, read it, cite | torque (40% part / 30% torque / 25% procedure / 5% pinout) |
| refusal | 20% | empty/insufficient context → refuse, don't invent | a real vehicle + a question with no answer in context |
| elicitation | 10% | no vehicle given → ask for VIN/engine/model+year | vehicle-specific Q with no `[VEHICLE]` block |
| cross_vehicle | 10% | this-car miss → surface a real shared-engine sibling, flag verify | `epc_parts` + `epc_section_profiles` + `profiles` |
| multi_turn | 10% | sticky vehicle across turns; NO topic bleed | two different-topic docs for the same model |
| figure | 4% | "show me the figure for X" (retrieval only) | docs with real `image_refs` |

## Honesty constraints (DECISIONS)
- **D5 figures = retrieval only.** No callout→part-number mapping exists in the data;
  the figure answer explicitly says it cannot label balloons. We never fabricate one.
- **D6 torque is parsed, not assumed.** `dataset/torque.py` extracts spec-table +
  inline-prose values, discards "see VCC-…" stubs, and reports MEASURED precision.
- **D4 cross-vehicle is honest + fallback-only.** A record is emitted ONLY when a real
  sibling vehicle genuinely shares the related vehicle's engine and the part does not
  list the active model — a true miss with a real shared basis. No fabricated
  vehicle/engine pairs. Never a supersession claim (that data isn't present).
- **D3 sticky vehicle, independent topic.** Turn 1 sets the car; turn 2 omits the
  `[VEHICLE]` block (inherited) and queries the new topic with its own words only.

## Splits / leakage
`split_key = "<model>|<topic>"`. The frozen benchmark
(`vida-eval/frozen_split.json`) reserves split_keys that `gen_dataset.py` EXCLUDES, so
the eval set is never trained on (D16). VIDA rows duplicate across platform-mates, so
splitting is by vehicle+topic, never random.

## Reproduce
```
python -m dataset.gen_dataset --limit 5000 --out data/ft/train.jsonl --seed 3407
```
Deterministic (ordered sources + seed). Prints exact per-type counts + any shortfall
(no silent caps). Re-runnable; idempotent for a fixed seed + DB.
