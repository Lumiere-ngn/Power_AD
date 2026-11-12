# Hand-specified graph, v0 (draft for BusPas review)

Built 9 November 2026. Not frozen: it gets frozen (dated commit) after the BusPas review, before any test-period run.

## Files

- `edges.csv`: every edge with its kind, sign, lag, reason, main source and BusPas question.
  - **E** edges (30) are drawn.
  - **P** edges (26) are pending: the documentation supports them, but they depend on how the sign is built or wired.
  - **X** edges (4) are not drawn, and each row says why.
- `nodes.csv`: the 22 measured and 14 context nodes, with the raw metric behind each.
- `QUESTIONS_BUSPAS.md`: 15 questions for one meeting with a BusPas engineer. Q1 matters most.
- `sources.json`: for each edge, the quotes the checking agents confirmed, plus all 361 quoted facts from the research step.

## How it was built

1. **Nodes** come from the dataset README: what each sign reports. No telemetry values, correlations, the pilot graph or the slide deck were used.
2. **Research.** Nine agents read one source each and pulled out quoted facts:
   - the NVIDIA Jetson Linux power and thermal pages
   - the Thermal Design Guide and the module Design Guide
   - the TI INA238 and INA3221 datasheets
   - DeepStream and LTE modem documents
   - solar charge-controller manuals
   - the README definitions
   - physics and application notes
3. **Drafts.** Three agents each drafted the edges on their own from those facts. Of the 60 distinct edges they proposed, 53 came from two or more drafters.
4. **Checks.** Each edge was checked twice, once against its sources and once for its mechanism. When the two checks disagreed (39 edges), a third agent decided. A completeness critic then added one edge (E18).
5. **Editor's changes.** These are listed in the `note` column:
   - The solar load triple is written as I = P / V, which removed a cycle.
   - `wwan_tx_rate -> input_power_watts` moved to pending, because it depends on the same wiring question as `wwan_rx_rate`.
   - One edge kind was relabelled.

## Conventions

- **Context nodes** are inputs only: they have no parents and are never checked.
- **Power = voltage × current:** current is the child (I = P / V). Power then stays the node that workload explains.
- **Lags:** within one hour the graph is acyclic. The only lagged edges (lag 1 h) are battery charge and discharge acting on battery voltage.
- **No fans:** cooling is passive. Throttling at 99 °C, power-limit throttling, under-voltage and leakage are documented feedback loops. They are not drawn, because they would close a cycle within the hour (see the BusPas questions).
- **Unobserved loads** (heater, display, throttling state) are not nodes.

## Power and heat (all signs)

Solid = drawn, dashed = pending BusPas, blue = context.

```mermaid
flowchart LR
  supply_56v -->|E01| input_voltage_volts
  input_power_watts -->|E02| input_current_amperes
  input_voltage_volts -->|E03| input_current_amperes
  input_power_watts -->|E04| input_energy_rate_w
  cpu_gpu_power_watts -->|E05| orin_power_watts
  deepstream_running -->|E06| cpu_gpu_power_watts
  deepstream_running -->|E07| orin_power_watts
  cpu_gpu_power_watts -->|E08| cpu_temp_celsius
  orin_power_watts -->|E09| cpu_temp_celsius
  enclosure_temperature -->|E10| cpu_temp_celsius
  cpu_temp_celsius -->|E11| gpu_temp_celsius
  deepstream_running -->|E12| gpu_temp_celsius
  enclosure_temperature -->|E13| board_temp_celsius
  input_power_watts -->|E14| board_temp_celsius
  orin_power_watts -.->|P01| input_power_watts
  wwan_tx_rate -.->|P02| input_power_watts
  wwan_rx_rate -.->|P03| input_power_watts
  wifi_hotspot_running -.->|P04| input_power_watts
  supply_56v -.->|P05| input_power_watts
  enclosure_temperature -.->|P06| input_power_watts
  humidity_pct -.->|P07| input_power_watts
  light_lux -.->|P08| input_power_watts
  orin_power_watts -.->|P09| device_power_watts
  light_lux -.->|P23| board_temp_celsius
  classDef ctx fill:#eef,stroke:#88a,color:#000
  class deepstream_running,enclosure_temperature,humidity_pct,light_lux,supply_56v,wifi_hotspot_running,wwan_rx_rate,wwan_tx_rate ctx
```

## Solar block (13 signs)

```mermaid
flowchart LR
  time_of_day -->|E15| mppt_solar_voltage
  time_of_day -->|E16| mppt_solar_current
  mppt_solar_voltage -->|E17| mppt_solar_current
  mppt_battery_voltage -->|E18| mppt_solar_voltage
  mppt_battery_voltage -->|E19| mppt_solar_current
  mppt_load_power -->|E20| mppt_solar_current
  mppt_charge_power -->|E21| mppt_battery_charge_current
  mppt_battery_voltage -->|E22| mppt_battery_charge_current
  mppt_battery_charge_current -->|E23 lag 1| mppt_battery_voltage
  mppt_load_current -->|E24 lag 1| mppt_battery_voltage
  mppt_battery_voltage -->|E25| mppt_battery_soc_pct
  mppt_battery_voltage -->|E26| mppt_load_voltage
  mppt_load_state -->|E27| mppt_load_power
  mppt_load_power -->|E28| mppt_load_current
  mppt_load_voltage -->|E29| mppt_load_current
  mppt_charge_power -->|E30| mppt_controller_temperature
  battery_soc_pct -.->|P10| input_voltage_volts
  mppt_battery_soc_pct -.->|P11| battery_soc_pct
  mppt_battery_voltage -.->|P12| input_voltage_volts
  mppt_load_voltage -.->|P13| input_voltage_volts
  input_current_amperes -.->|P14| mppt_load_current
  input_power_watts -.->|P15| mppt_load_power
  mppt_solar_voltage -.->|P16| mppt_charge_power
  mppt_solar_current -.->|P17| mppt_charge_power
  mppt_load_state -.->|P18| mppt_load_voltage
  mppt_battery_temperature -.->|P19| mppt_solar_current
  mppt_battery_temperature -.->|P20| mppt_battery_voltage
  light_lux -.->|P21| mppt_solar_current
  light_lux -.->|P22| mppt_solar_voltage
  light_lux -.->|P24| mppt_battery_temperature
  enclosure_temperature -.->|P25| mppt_controller_temperature
  enclosure_temperature -.->|P26| mppt_battery_temperature
  classDef ctx fill:#eef,stroke:#88a,color:#000
  class enclosure_temperature,light_lux,mppt_load_state,time_of_day ctx
```

## Known gaps

- **`input_power_watts` has no drawn parent.** The main energy node waits on Q1: does the module, modem, display and heater power pass through the INA238?
- **No edges yet** for `device_power_watts` (undefined), `battery_soc_pct` and `mppt_battery_temperature`.
- **`mppt_charge_power` has no parents.** It is unclear whether it is measured on the panel side or the battery side (Q14).
- **Weakest drawn edges:**
  - E10 and E13 assume the environment sensor shares air with the module and the INA238 (Q13).
  - No single source states E12 end to end.
  - E28 and E29 rest on one controller family's register map.
- **Clean before use:**
  - `gpu_temp_celsius = -256` (GPU power-gated) is missing data, not a temperature.
  - The MPPT placeholder values of exactly 100 must be removed.
  - Hours that contain an energy-counter reset break E04.
