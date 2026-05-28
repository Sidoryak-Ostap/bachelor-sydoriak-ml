import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional


app = FastAPI(
    title="AgroMap ML Engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["localhost", "http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

try:
    model_pipeline = joblib.load('yield_forecast_pipeline.pkl')
    print("[ОК] Пайплайн моделі успішно завантажено в пам'ять сервера.")
except Exception as e:
    print(f"[ПОМИЛКА] Не вдалося завантажити файл моделі: {e}")
    model_pipeline = None


df_historical = pd.read_csv('agro_dataset.csv')
baseline_stats = df_historical.groupby(['crop_type', 'soil_type']).mean().to_dict(orient='index')


class FieldData(BaseModel):
    soil_type: str
    crop_type: str
    ndvi_early: Optional[float] = None
    rain_sum_early: Optional[float] = None
    temp_sum_early: Optional[float] = None
    
    ndvi_mid: Optional[float] = None
    rain_sum_mid: Optional[float] = None
    temp_sum_mid: Optional[float] = None
    
    ndvi_late: Optional[float] = None
    rain_sum_late: Optional[float] = None
    temp_sum_late: Optional[float] = None

@app.get("/")
def read_root():
    return {"status": "healthy", "model_loaded": model_pipeline is not None}

@app.post("/predict-yield")
def predict_yield(payload: FieldData):
    try:
        input_dict = payload.model_dump()
        
        lookup_key = (payload.crop_type, payload.soil_type)
        if lookup_key not in baseline_stats:
            raise HTTPException(status_code=400, detail="Нетипова комбінація культури та ґрунту")
            
        historical_defaults = baseline_stats[lookup_key]
        
        has_early = payload.ndvi_early is not None
        has_mid = payload.ndvi_mid is not None
        has_late = payload.ndvi_late is not None

        if has_early and has_mid and has_late:
            forecast_status = "final"
            confidence_level = "high"
        elif has_early and has_mid:
            forecast_status = "mid_preliminary"
            confidence_level = "medium"
        else:
            forecast_status = "early_preliminary"
            confidence_level = "low"
            
        for key in input_dict.keys():
            if input_dict[key] is None:
                input_dict[key] = historical_defaults[key]
                
        input_df = pd.DataFrame([input_dict])
        prediction = model_pipeline.predict(input_df)[0]
        
        return {
            "success": True,
            "predicted_yield": round(float(prediction), 2),
            "unit": "т/га",
            "meta": {
                "status": forecast_status,         
                "confidence": confidence_level,    
                "message": "Прогноз частково базується на середніх історичних показниках клімату." if forecast_status != "final" else "Прогноз фінальний."
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))