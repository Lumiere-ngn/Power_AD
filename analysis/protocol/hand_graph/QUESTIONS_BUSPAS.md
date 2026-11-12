# Questions for BusPas about the hand graph

These are for one meeting with a BusPas engineer. Each answer adds, removes or confirms edges in `edges.csv` (E = drawn, P = pending). Q1 matters most.

## Q1. Power path (highest priority)

Which loads draw current through the INA238 shunt that reports device_input_voltage/current/power? Is that shunt on the sign's main input, upstream of the converter that makes the Orin module's 5 V VDD_IN? Is it also upstream of the display/backlight, LTE modem, Wi-Fi card, camera, heater and the PIR, light and environment sensors? Does any load bypass it? Which converter(s) feed the module? Can a sag at the sign input ever pull the module's 5 V rail down to NVIDIA's ~4.5 V under-voltage (OC1) trigger?

*Why it matters:* This one answer decides about 8 edges. Until it is answered, input_power_watts has no drawn parent, so the largest load (the module) is not modelled. The containment orin_power -> input_power is valid only if the module sits behind the shunt. Any load that bypasses the shunt must lose its edge into input_power. The under-voltage answer decides whether input_voltage_volts has any control path to throttling.

*Edges:* E14, P01, P02, P03, P04, P06, P07, P08

## Q2. Power path

56 V signs and hardware flags: (1) What feeds the ~56 V signs: PoE, a 48 V telecom plant or a mains PSU? (2) What sits between their input and the electronics: an extra 56->12 V stage feeding the same 12 V build, a single 56->5 V converter, or a different build with different loads? (3) Is the INA238 upstream of every converter on both builds, and is the shunt or board the same? (4) Do any 56 V signs have a battery or charge controller? (5) Are supply_56v and has_solar taken from hardware records rather than inferred from voltage telemetry? Please send the per-device list.

*Why it matters:* If 56 V signs carry an extra conversion stage inside the INA238 path, supply_56v becomes a documented parent of input_power_watts. If the flags were derived from telemetry, two context roots would be data-derived, which breaks the knowledge-based rule.

*Edges:* E01, P05

## Q3. Power path

Solar sign wiring: on the 13 solar signs, where is the INA238 input connected: (a) to the charge controller's LOAD+/LOAD- terminals, (b) directly to the battery terminals or fuse block, or (c) through a DC-DC converter or the power-manager board? Is anything else on the load output that bypasses the INA238, such as the heater, lighting or display? Is there a fuse, e-fuse or diode in front of the INA238? Does mppt_load_state = 0 switch off the sign itself, so that a load-off hour shows up as a telemetry gap?

*Why it matters:* These edges are the only cross-check between the INA238 and the charge controller. On the load output, load_voltage ~ input_voltage and load_power ~ input_power plus wiring loss. On the battery, input_voltage ~ battery_voltage. Behind a converter, neither redundancy holds.

*Edges:* E24, P12, P13, P14, P15, P18

## Q4. Telemetry definitions

INA238 readings: (1) Does the agent read input voltage, current and power straight from the hwmon files (in1_input, curr1_input, power1_input), or does it compute one from the other two (P = V x I or I = P/V)? (2) Which driver provides sysfs:ina238 on the Jetson kernel: a backport, a vendor driver or userspace I2C? (3) Which shunt resistance, shunt gain and ADC averaging/conversion time are set, and are they the same on all ~86 devices and on the 12 V and 56 V boards? (4) Can current through the shunt reverse, for example while a battery charges? (5) Can the INA238 lose power independently of the Jetson, which would leave its calibration at the reset value?

*Why it matters:* An identity edge on a value the agent computes from its own parents has a zero residual by construction and cannot detect anything. Only identities between independently measured registers are informative. Signed versus unsigned power, and shunt values that differ between devices, change whether the identity holds exactly.

*Edges:* E02, E03

## Q5. Telemetry definitions

Energy counter and time: (1) How is device_input_energy_wh_total computed? Which power value does it integrate (the INA238 POWER register or V x I), at what time step, by rectangle or trapezoid rule, and does it run independently of the 120 s scrape? (2) When does it reset: at local midnight, at agent restarts, at reboots? (3) Is the unit really Wh, given the README's typical maximum of 18,891? (4) Is time_of_day meant as local time? Which time zone does each agency use (the analysis currently builds UTC hours)?

*Why it matters:* The energy identity holds only if the counter integrates the same power at a fine step and resets can be detected. 18,891 Wh is far more than a day at 10-18 W, so the unit or the reset behaviour is in doubt. The solar edges assume local solar time.

*Edges:* E04, E15, E16

## Q6. Telemetry definitions

Module rails and heartbeat metrics: (1) Which INA3221 channel does the agent read for device_orin_power_watts: ch1 VDD_IN, ch2 VDD_CPU_GPU_CV, ch3 VDD_SOC, or a sum? Is it in_input x curr_input from single samples, or a tegrastats/jtop value, instantaneous or averaged? (2) Where do the heartbeat's cpu_gpu_power and device_power come from: which tool, which rail, instantaneous or averaged, and how often are they written? Can they be stale relative to the scrape? (3) What is device_power, and why is its median about 5 W against about 10 W input? (4) What do the INA3221 samples and update_interval attributes read?

*Why it matters:* The containment cpu_gpu <= orin is valid only if orin_power is ch1 VDD_IN and cpu_gpu is ch2. If device_power is the same rail, it gets a redundancy edge; otherwise it stays parentless and effectively unchecked. A mismatch in averaging windows between the two tools adds noise to every edge between them.

*Edges:* E05, E07, E08, E09, P01, P09

## Q7. Compute configuration

(1) Which module SKU does each device run: Orin Nano 8GB, Orin Nano 4GB or Orin NX? The README RAM totals are 7.3 and 3.4 GB. Is the Super flash configuration used? (2) Which nvpmodel mode does each device run, did it change during the dataset period, is there a custom nvpmodel.conf, and is jetson_clocks run at boot? (3) Which JetPack/L4T release is installed? (4) Do signs ever enter SC7 suspend or power the module down on a schedule, for example at night or on low battery?

*Why it matters:* SKU and power mode set the VDD_IN cap (10, 15 or 25 W) and the GPU size, so the same DeepStream load maps to different power. Since no SKU node exists, this needs per-device stratification. The L4T release decides which NVIDIA doc version applies. Suspend or a scheduled power-down drives module power toward zero regardless of the load nodes.

*Edges:* E05, E06, E07, E09

## Q8. Thermal configuration and observability

(1) Which thermal zone feeds cpu_temp_celsius: cpu-thermal or tj-thermal? (2) What does the agent record when a gpu-thermal read fails with -EAGAIN while the GPU is power-gated: -256, 0, NaN, the last value, or nothing? (3) Were trip points, governors or the INA3221 OC limits (curr*_crit/max) changed from NVIDIA defaults for the fanless enclosure? The defaults are 99.0 C software throttle, 103 C hardware throttle and 104.5/105 C shutdown. (4) Do the 11 signs that cluster near 99 C throttle continuously, and is that accepted by design? (5) After a 105 C hardware shutdown, does the carrier power the module back on? (6) Could the agent export cooling_device*/cur_state, oc*_event_cnt and CPU/GPU clock frequencies?

*Why it matters:* Placeholder values such as -256 from a gated GPU would corrupt the hourly gpu_temp means and break the redundancy edge. Changed trip points move the saturation regime. An exported throttle state would turn the unobserved lag-0 control loop into a labelled regime, so 99 C hours would not be flagged as anomalies.

*Edges:* E08, E09, E11, E12

## Q9. Workload

(1) When DeepStream runs, what does the pipeline do? Please give the camera source (CSI, USB, RTSP or PoE) and whether the camera is powered from the sign input; the number of streams, resolution and fps; whether inference runs on the GPU via nvinfer/TensorRT the whole time the process is up; NVDEC/VIC use; any software encode; tracker and OSD; and whether it publishes over LTE (nvmsgbroker, MQTT, Kafka). (2) Why is it usually not running? (3) What is a 'DeepStream consumer error', and do these errors coincide with pipeline restarts? (4) Is there any other significant GPU workload, such as display rendering?

*Why it matters:* GPU inference is what makes deepstream_running a parent of cpu_gpu_power. NVDEC/VIC load on VDD_SOC justifies its edge into orin_power beyond the CPU and GPU. Publishing over LTE confounds deepstream with wwan_tx. If errors mean restarts, ds_error_rate could become a documented parent with a negative effect on load.

*Edges:* E06, E07, E12

## Q10. Peripherals

LTE and Wi-Fi: (1) Which LTE modem is fitted (make, model, Cat class)? How is it connected (M.2 Key B, mini-PCIe, USB or an external router), and from which rail is it powered? (2) Do wwan_rx/tx count all bytes on the modem interface (e.g. wwan0 rx_bytes/tx_bytes), including hotspot and DeepStream traffic? (3) Is any radio metric (RSRP, RSRQ, SINR or TX power) logged? (4) Which Wi-Fi module provides the hotspot? Does wifi_hotspot_running = 1 mean an access point is actually running? When it is 0, is the radio off or rfkilled, in station mode, or not fitted? (5) Does the hotspot tether over LTE?

*Why it matters:* The radio edges need the radio behind the INA238. Byte counts stand in for radio-active time, and without TX power the path loss is an unobserved driver of modem power. Tethering would give wifi and wwan a shared cause.

*Edges:* P02, P03, P04

## Q11. Peripherals

Display: (1) What technology is it: backlit LCD, LED matrix, or e-paper with a front light? (2) Is its brightness set from the light sensor, by a time-of-day schedule, or woken by PIR motion? (3) Roughly how many watts does it swing between darkest and brightest, and does it have its own heater? (4) Is it powered behind the INA238?

*Why it matters:* The display is probably the largest switched load outside the module. Without a documented brightness law, no lux, PIR or time edge into input_power can be cited, and normal dimming swings would be flagged as anomalies.

*Edges:* P08

## Q12. Peripherals

Heater: (1) Is a heater fitted on every sign that has the heaterenv sensor (about 64)? (2) What switches it: a hardware thermostat, a hygrostat, both, or the heaterenv software? On which reading (temperature, humidity or both), and at what set points and hysteresis? (3) What is its wattage? (4) Is it powered behind the INA238, from the MPPT load output, or from a separate feed? (5) Is its on/off state logged anywhere?

*Why it matters:* In cold hours the heater may be the largest single load. A cited control law gives a threshold-shaped edge from the heaterenv readings into input power. A logged on/off state would remove an unobserved driver.

*Edges:* P06, P07

## Q13. Physical layout

Where are the following mounted relative to each other? (1) The heaterenv temperature/humidity sensor: in the internal air, near the heater, near the Orin heat sink, or outside? (2) The Orin's passive heat sink: does it shed heat into that air, or is it bonded to the enclosure wall? (3) The INA238 board: near the DC-DC converter, the module or the input connector? (4) The light sensor: which chip and gain; facing the sky or behind display or tinted glass; does it see the same sky as the solar panel; does it saturate in sun? (5) The charge controller and battery: in the same air space as the heaterenv sensor, in a separate or sun-exposed box, or at the pole base?

*Why it matters:* Ambient-temperature edges need a shared air volume. If the heaterenv sensor is not the heat sink's ambient, the drawn enclosure -> cpu_temp and board_temp edges are wrong. light_lux can replace time_of_day as the irradiance parent of the solar nodes only if it sees the panel's sky.

*Edges:* E10, E13, E14, E15, P22, P23, P24, P25, P26

## Q14. Solar charge controller

(1) Which controller (make, model, firmware) is fitted, and which Modbus register feeds each sqlite:mppt field? (2) Is mppt_charge_power the PV-side 'charging power' (SRNE/Renogy 0x0109 = solar V x solar I) or battery-side power (e.g. EPEVER 0x3106)? (3) Is mppt_battery_charge_current the converter's total output, part of which feeds the load, or the net battery current? (4) Is mppt_load_power read from a register or computed as V x I? (5) Is mppt_load_state the read-back status or the commanded value, and does load voltage read about 0 V when the output is off? (6) Which load mode (manual/always on, light control or timer) and which LVD/LVR and PV start/stop set points are configured, and does BusPas firmware ever write the load register? (7) At what cadence is the controller polled? Do any fields use placeholder codes such as exactly 100?

*Why it matters:* PV-side versus battery-side decides whether charge_power is an identity child of solar V and I or of battery V and charge current. Gross versus net decides whether load current enters charge current. LVD and light control make load_state endogenous. The model is needed to cite the controller's own thresholds and register definitions.

*Edges:* E17, E18, E19, E20, E21, E26, E28, P16, P17, P18, X03, X04

## Q15. Battery and state of charge

(1) What battery chemistry and capacity are fitted: flooded, AGM or gel lead-acid, or LiFePO4 with a BMS? Does a BMS send a charge limit to the controller? (2) Is a battery temperature probe fitted on each solar sign, or does mppt_battery_temperature fall back to a 25 C default or the internal sensor? (3) Is temperature compensation on, with what coefficient, and what are the low and high battery-temperature cut-offs? (4) Does the agent decode temperatures as sign-magnitude, i.e. are the 129-131 C readings really -1 to -3 C? (5) Does the controller compute SOC from voltage only or by coulomb counting? Which of mppt_battery_remaining_capacity and its _mppt variant is authoritative, and why are there 17 devices versus 13? (6) Where does the power manager's battery_soc_pct come from (a BMS, a copy from the controller, a voltage formula or a coulomb counter), and which battery does it describe? How are the roughly 27 non-solar battery signs wired and charged?

*Why it matters:* A voltage-only SOC makes the redundancy edge nearly exact; coulomb counting makes it weaker. Temperature compensation and cut-offs are the only documented routes for battery temperature into voltage and charging. An undecoded sign bit would make every sub-zero hour look like an anomaly.

*Edges:* E25, P11, P19, P20, X01
