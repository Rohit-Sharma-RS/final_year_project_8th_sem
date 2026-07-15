# Complete Project Workflow: Carbon-Aware Routing

## 🎯 Project Goal
Find routes with the **least carbon footprint** considering:
- Traffic congestion (vehicle count, speed, vehicle types)
- Building density vs vegetation 
- Wind direction and speed
- Air Quality Index (AQI)
- Distance and time

---

## 📊 Phase 1: Data Fusion ✅ (COMPLETED)

**File**: `pipeline.py`

### What you already have:
1. **Road Network**: 90,915 road segments from OSM
2. **Building Footprints**: 191,350 buildings
3. **Vegetation**: 898 parks/forests/green spaces
4. **Weather**: Real-time wind speed & direction
5. **AQI**: Real-time air quality via OpenWeatherMap
6. **Traffic**: Currently simulated (will be replaced by SUMO)

### Output:
- `fused_roads.geojson` - Geographic data
- `fused_roads.csv` - Tabular data


---

## 🚗 Phase 2: SUMO Traffic Simulation (NEXT STEP)

**File**: `sumo_integration.py`

### What SUMO adds:
- **Realistic traffic flow** (not random numbers)
- **Multiple vehicle types**:
  - Petrol cars → High emissions
  - Diesel cars → Moderate emissions  
  - Electric cars → Zero emissions
  - Buses, motorcycles, trucks
- **Time-based patterns**:
  - Rush hour (7-10 AM, 5-8 PM)
  - Midday, night traffic
- **Real CO2 emissions per vehicle**

### Steps:
```powershell
# 1. Install SUMO
winget install DLR.SUMO

# 2. Install Python interface
pip install traci eclipse-sumo sumolib

# 3. Run integration script
python sumo_integration.py
```

### What it does:
1. Exports Kolkata roads to OSM format
2. Converts to SUMO network (`kolkata.net.xml`)
3. Generates traffic with vehicle type mix
4. Runs simulation for 1 hour (configurable)
5. Collects real-time data:
   - Vehicle count per road
   - Average speed per road
   - CO2 emissions per road
   - Congestion levels
6. Merges with your fused dataset

### Output:
- `fused_roads_with_sumo.geojson` - Enhanced dataset
- `sumo_traffic_data.csv` - Raw SUMO results

---

## 🗺️ Phase 3: Eco-Routing (READY TO USE)

**File**: `eco_routing.py`

### Routing Strategies:

#### 1. **Shortest Distance**
Traditional GPS routing - minimize kilometers

#### 2. **Fastest Time**
Current apps like Google Maps - minimize minutes

#### 3. **Lowest Carbon** ⭐
New approach - minimize environmental impact

#### 4. **Balanced**
Mix of all factors:
- 50% carbon cost
- 30% time
- 20% distance

### Usage:
```python
from eco_routing import compare_routes

# Find origin/destination node IDs from your dataset
origin_id = 123456789
destination_id = 987654321

# Compare all routing strategies
results = compare_routes(origin_id, destination_id)
```

### Example Output:
```
📍 Shortest Distance
   Distance: 8.5 km
   Time: 15.2 minutes
   Carbon Footprint: 1250.50
   
📍 Fastest Time
   Distance: 10.2 km
   Time: 12.8 minutes
   Carbon Footprint: 1580.30
   
📍 Lowest Carbon ⭐
   Distance: 9.1 km
   Time: 16.5 minutes
   Carbon Footprint: 980.20
   
📍 Balanced
   Distance: 8.9 km
   Time: 14.8 minutes
   Carbon Footprint: 1100.40

🌱 RECOMMENDED ECO-ROUTE: Lowest Carbon
   Saves 270.30 carbon units vs shortest route
```

---

## 📁 Complete File Structure

```
proj/
├── pipeline.py                      # ✅ Data fusion (done)
├── eco_routing.py                   # ✅ Routing algorithms (ready)
├── sumo_integration.py              # 🔄 Traffic simulation (next)
├── PIPELINE_DOCUMENTATION.md        # ✅ Documentation
│
├── fused_roads.geojson              # ✅ Your current dataset
├── fused_roads.csv                  # ✅ Your current dataset
│
├── kolkata_roads.osm                # 🔄 Will be created by SUMO
├── kolkata.net.xml                  # 🔄 SUMO network
├── traffic.rou.xml                  # 🔄 SUMO routes
├── kolkata.sumocfg                  # 🔄 SUMO config
├── emissions.xml                    # 🔄 SUMO output
├── sumo_traffic_data.csv            # 🔄 SUMO results
├── fused_roads_with_sumo.geojson    # 🔄 Enhanced dataset
│
└── cache/                           # OSM cache
```

---

## 🚀 Next Steps - WHAT TO DO NOW

### Step 1: Install SUMO
```powershell
# Windows
winget install DLR.SUMO

# Add to PATH or use full path:
# C:\Program Files (x86)\Eclipse\Sumo\bin\
```

### Step 2: Install Python packages
```powershell
pip install traci eclipse-sumo sumolib
```

### Step 3: Run SUMO integration
```powershell
python sumo_integration.py
```

This will:
- Create SUMO network files
- Generate traffic patterns
- Run simulation (opens GUI)
- Collect real traffic data
- Update your dataset

### Step 4: Test eco-routing
```python
python eco_routing.py
```

### Step 5: Compare results
```python
import pandas as pd

# Old (simulated traffic)
old = pd.read_csv("fused_roads.csv")
print(f"Simulated avg vehicle count: {old['vehicle_count'].mean()}")

# New (SUMO traffic)
new = pd.read_csv("fused_roads_with_sumo.csv")
print(f"SUMO avg vehicle count: {new['avg_vehicle_count'].mean()}")
```

---

## 🎓 How Simulation Works

### Without SUMO (Current):
```python
# Random traffic - not realistic
edges["vehicle_count"] = np.random.randint(20, 200, len(edges))
edges["avg_speed"] = np.random.randint(20, 60, len(edges))
```

### With SUMO (After integration):
```python
# Real traffic simulation
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()  # 1 second of real traffic
    
    for road in roads:
        # Each vehicle moves based on:
        # - Speed limits
        # - Traffic lights
        # - Other vehicles (car-following model)
        # - Lane changes
        # - Acceleration/deceleration
        
        vehicle_count = traci.edge.getLastStepVehicleNumber(road)
        avg_speed = traci.edge.getLastStepMeanSpeed(road)
        co2 = traci.edge.getCO2Emission(road)
```

---

## 📈 Expected Improvements with SUMO

| Metric | Before (Random) | After (SUMO) |
|--------|-----------------|--------------|
| **Traffic realism** | Random 20-200 vehicles | Realistic flow patterns |
| **Speed accuracy** | Random 20-60 km/h | Based on congestion |
| **Vehicle types** | Generic | Petrol/Diesel/Electric/Bus/Truck |
| **Time patterns** | Static | Rush hour, midday, night |
| **CO2 emissions** | Estimated | Measured per vehicle type |
| **Route quality** | Good | Excellent ⭐ |

---

## 🔬 Research Applications

1. **Urban Planning**: Identify high-emission corridors
2. **Policy Making**: Test impact of green zones on emissions
3. **Traffic Management**: Optimize signal timing for lower emissions
4. **Public Transport**: Design eco-friendly bus routes
5. **EV Adoption**: Model impact of electric vehicle penetration
6. **Route Optimization**: Build carbon-aware navigation apps

---

## 💡 Future Enhancements

### Short term (After SUMO):
- [ ] Web dashboard with map visualization
- [ ] Mobile app for route comparison
- [ ] Real-time route updates based on traffic

### Medium term:
- [ ] Machine learning to predict emissions
- [ ] Integration with Google Directions API
- [ ] Multi-city support (Mumbai, Delhi, etc.)

### Long term:
- [ ] Commercial navigation app
- [ ] Partnership with ride-sharing services
- [ ] Government policy recommendations

---

## ❓ FAQ

**Q: Why SUMO instead of just random traffic?**
A: SUMO simulates realistic vehicle movement including congestion, traffic lights, lane changes. Random numbers don't capture real-world traffic patterns.

**Q: How long does SUMO simulation take?**
A: ~5-10 minutes for 1 hour of simulated traffic on 90K roads.

**Q: Can I use real traffic data instead of SUMO?**
A: Yes! If you have access to Google Traffic API, TomTom, or city traffic sensors, you can use that instead.

**Q: What if origin/destination nodes not found?**
A: Use `ox.distance.nearest_nodes()` to find closest nodes to lat/long coordinates.

**Q: How to visualize routes?**
A: Use `folium` library to create interactive maps or load GeoJSON in QGIS.

---

## 📞 Commands Quick Reference

```powershell
# Run data fusion
python pipeline.py

# Run SUMO integration
python sumo_integration.py

# Test eco-routing
python eco_routing.py

# Manual SUMO simulation
sumo-gui -c kolkata.sumocfg

# View results
python -c "import pandas as pd; print(pd.read_csv('fused_roads_with_sumo.csv').head())"
```

---

**Status**: Phase 1 ✅ Complete | Phase 2 🔄 Ready to start | Phase 3 ✅ Ready to use
