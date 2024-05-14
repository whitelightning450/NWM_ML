![NWM_ML](./Images/NWM_ML_Hydrological_Cycle.png)

# NWM-ML
A Machine Learning extension processor coupling the NWM to water resources management tools in the western US.
The testing location of the prototype NWM-ML is in the Great Salt Lake (GSL) basin with a motivation to extend the 30-day NWM operational product to a season-to-season water supply forecasting tool to support water resources managment.
The trained models will be coupled to the GSL Integrated model wetlands and lake modules to demonstrate a proof of concept water resources management tool for the GSL - a tool to address the beneficial use of environmental management, agriculture, and a growing urban population.
Upon the successful demonstration of the NWM-ML in the GSL basin, model development will scale to the Upper Colorado River Basin.


The goal of the machine learning model workflow is to correct the NOAA National Water Model (NWM) streamflow predictions impacted by extensive water resource infrastructure in the Western US.
This incluces but is not limited to reservoirs, diversions, interbasin transfers, municipal water demands, and routine undocumented water use.
The workflow leverages the power of the physically-based NWM v2.1 retrospective (soon to be NWM v3.0) and integrates key catchment characteristics including:

**StreamStats**
* Drainage area (mi2)
* Mean Basin Elevation (ft)
* Land Cover - Percent forest, percent developed, percent impervious, percent herbacious
* Basin slope - Percent of catchment with a slope greater than 30 degrees

**NWM**
*v2.1 retrospective flows resampled to a daily temporal resolution

**Seasonality Metrics**
*S1
*S2

**United States Bureau of Reclamation**
* Upstream catchment storage - Percent of full capacity

**National Resource Conservation Service**
* snow-water-equivalent -  Average catchment snow-water-equivalent provided by Snow Telemetry (SNOTEL) monitoring sites

**National Land Data Assimilation System**
* Daily precipitation (in)
* Mean annual precipitation (in)
* Daily temperature (F)

**NOAA Analysis of Record for Calibration**
* Daily precipitation (in) - replacing NLDAS
* Mean annual precipitation (in)- replacing NLDAS
* Daily temperature (F) - replacing NLDAS

**United States Geological Survey**
* National Water Information System (NWIS) streamflow monitoring informatin for colocated NHDPlus reaches for training/testing targets



Funding for this project was provided by the National Oceanic and Atmospheric Administration (NOAA), and awarded to the Cooperative Institute for Research to Operations in Hydrology (CIROH) through the NOAA Cooperative Agreement with The University of Alabama, NA22NWS4320003.


