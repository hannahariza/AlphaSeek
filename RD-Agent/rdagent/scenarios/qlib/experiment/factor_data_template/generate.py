import os
import qlib

# Ensure we use standard qlib data path, not QuantaAlpha's
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")

from qlib.data import D

instruments = D.instruments()
fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
data = D.features(instruments, fields, freq="day").swaplevel().sort_index().loc["2008-12-29":].sort_index()

data.to_hdf("./daily_pv_all.h5", key="data")


# For debug data, select first 100 available instruments from the data
try:
    fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
    debug_data = (
        D.features(instruments, fields, start_time="2018-01-01", end_time="2019-12-31", freq="day")
        .swaplevel()
        .sort_index()
    )
    
    # Get available instruments from the data itself
    available_instruments = debug_data.reset_index()["instrument"].unique()
    selected_instruments = available_instruments[:min(100, len(available_instruments))]
    
    # Filter data to selected instruments
    debug_data = (
        debug_data
        .swaplevel()
        .loc[selected_instruments]
        .swaplevel()
        .sort_index()
    )
    
    debug_data.to_hdf("./daily_pv_debug.h5", key="data")
    print(f"Successfully generated debug data with {len(selected_instruments)} instruments")
    
except Exception as e:
    print(f"Warning: Failed to generate debug data: {e}")
    # If debug data generation fails, just copy the full data
    data.to_hdf("./daily_pv_debug.h5", key="data")
    print("Using full data as debug data fallback")
