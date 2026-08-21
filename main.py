# Web dashboard address: 127.0.0.1:8000
# Interactive API control panel: 127.0.0.1:8000/docs
# username: admin_clinician
# password: password123

# Download library
from libAuto import libmain
libmain()

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import secrets
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import jwt
from pydantic import BaseModel, Field
# This imports code and loads the models automatically
from predict import predict_fall_risk, explain_patient, recommend_intervention
# This imports data generater
from generate_fall_risk_data import main
import gradio as gr
import pandas as pd
import random

from models import PatientRecord
from database import Base, engine, get_db, AsyncSessionLocal
from sqlalchemy.ext.asyncio import AsyncSession

from fastapi.middleware.cors import CORSMiddleware
from typing import Literal

SECRET_KEY = "SUPER_SECRET_SECURITY_KEY_CHANGE_THIS_IN_PRODUCTION"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# hashlib.sha256(b"password123").hexdigest()
ADMIN_PASSWORD_HASH = "ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f"

MOCK_USERS_DB = {
    "admin_clinician": {
        "username": "admin_clinician",
        "hashed_password": ADMIN_PASSWORD_HASH,
        "disabled": False,
    }
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class Token(BaseModel):
    access_token: str
    token_type: str

def verify_password(plain_password: str, correct_hash: str) -> bool:
    input_hash = hashlib.sha256(plain_password.encode('utf-8')).hexdigest()
    return secrets.compare_digest(input_hash, correct_hash)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Dependency validation check ensuring route protection
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
        
    user = MOCK_USERS_DB.get(username)
    if user is None:
        raise credentials_exception
    return user


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Shutdown
    await engine.dispose()

# Initialize the API
app = FastAPI(title="Fall Risk Assessment API", lifespan=lifespan)

# Define the 10 inputs exactly as requested in the README
class PatientData(BaseModel):
    sex: Literal['M', 'F']
    age: int = Field(..., ge=60, le=100)
    night_bed_exits: int = Field(..., ge=0, le=8)
    night_activity_duration_min: float = Field(..., ge=0, le=120)
    past_falls: int = Field(..., ge=0, le=5)
    mobility_score: int = Field(..., ge=1, le=10)
    high_risk_medication: int = Field(..., ge=0, le=1)
    cognitive_impairment: int = Field(..., ge=0, le=2)
    polypharmacy_count: int = Field(..., ge=0, le=14)
    orthostatic_hypotension: int = Field(..., ge=0, le=1)
    tug_seconds: float = Field(..., ge=8.0, le=31.9)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = MOCK_USERS_DB.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/predict")
async def get_prediction(data: PatientData, db: AsyncSession = Depends(get_db), current_user: dict = Depends(get_current_user)):
    try:
        # Convert incoming JSON data into a clean Python dictionary
        features_dict = data.model_dump()
        
        # Run model function
        result = predict_fall_risk(features_dict)
        lime_explanations = explain_patient(features_dict, max_features=3)
        suggestion = recommend_intervention(features_dict)

        # Save feature profile & prediction output to database
        db_record = PatientRecord(
            age=features_dict["age"],
            night_bed_exits=features_dict["night_bed_exits"],
            night_activity_duration_min=features_dict["night_activity_duration_min"],
            past_falls=features_dict["past_falls"],
            mobility_score=features_dict["mobility_score"],
            high_risk_medication=features_dict["high_risk_medication"],
            cognitive_impairment=features_dict["cognitive_impairment"],
            polypharmacy_count=features_dict["polypharmacy_count"],
            orthostatic_hypotension=features_dict["orthostatic_hypotension"],
            tug_seconds=features_dict["tug_seconds"],
            fall_risk_level=result
        )
        db.add(db_record)
        await db.commit()
        
        # Return the exact JSON structure your friend asked for
        return {
            "fall_risk_level": result,
            "lime_explanations": lime_explanations,
            "suggestion": suggestion
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
# Gradio Prediction Logic
async def predict_gradio(
        inputSex,
        inputAge, 
        inputPastFalls, 
        inputMobilityScore, 
        inputNightBedExits, 
        inputNightActivityDurationMin, 
        inputHighRiskMed, 
        inputCognitiveImpairment,
        inputPolypharmacyCount,
        inputOrthostaticHypotension,
        inputTugSeconds
    ):
    profile = PatientData(
        sex=str(inputSex),
        age=int(inputAge),
        night_bed_exits=int(inputNightBedExits),
        night_activity_duration_min=int(inputNightActivityDurationMin),
        past_falls=int(inputPastFalls),
        mobility_score=int(inputMobilityScore),
        high_risk_medication=1 if inputHighRiskMed else 0,
        cognitive_impairment=int(inputCognitiveImpairment), 
        polypharmacy_count=int(inputPolypharmacyCount), 
        orthostatic_hypotension=1 if inputOrthostaticHypotension else 0, 
        tug_seconds=float(inputTugSeconds)
    )

    async with AsyncSessionLocal() as session:
        response = await get_prediction(data=profile, db=session)
        
    risk_level = response["fall_risk_level"]
    explanations = response["lime_explanations"]
    suggestion = response["suggestion"]

    # Format output for the user interface textboxes
    output_text = f"Risk Level: {risk_level}\n"
    output_text += "primary factors:\n"
    for i, exp in enumerate(explanations, start=1):
        output_text += f" {i}. {exp['condition']} | Weight: {exp['weight']} ({exp['direction']})\n"
    output_text += f"Suggestions:\n"
    if risk_level == "HIGH":
        for item in suggestion["all_options"]:
            if item["can_flip"]:
                # Perfect Sentence Generation matching your UI requirements
                output_text += f"  • {item['feature']} need to change from {item['from']} to {item['to']}.\n"
            else:
                # Gracefully handle features that cannot drop the risk level independently
                output_text += f"  • [Restricted] Altering '{item['feature']}' alone is insufficient.\n"
    else:
        output_text += f"{suggestion["note"]}\n"
    return output_text

def random_gradio():
    main()
    df = pd.read_csv("fall_risk_patients_2000.csv")
    X = df.drop(columns=["fall_risk_score", "fall_risk_level", "patient_id", "name"])    # feature
    first_row = X.iloc[random.randint(0, 2000)].to_dict()
    return (
        first_row.get("age"),
        first_row.get("night_bed_exits"),
        first_row.get("night_activity_duration_min"),
        first_row.get("past_falls"),
        first_row.get("mobility_score"),
        first_row.get("high_risk_medication"),
        first_row.get("cognitive_impairment"),
        first_row.get("polypharmacy_count"),
        first_row.get("orthostatic_hypotension"),
        first_row.get("tug_seconds"),
    )

# Build Gradio UI
with gr.Blocks() as interface:
    gr.Markdown("# Gradio with FastAPI Fall Risk Predictor")
    with gr.Row():
        with gr.Column():
            inputSex = gr.Radio(
                choices=["M", "F"], 
                label="Sex",
                value="M"
            )
            inputAge = gr.Number(
                minimum=60, 
                maximum=100, 
                label="Age", 
                value=65
            )
            inputPastFalls = gr.Number(
                minimum=0, 
                label="Past falls", 
                value=0
            )
            inputMobilityScore = gr.Slider(
                minimum=1, 
                maximum=10, 
                step=1, 
                label="Mobility score", 
                value=5
            )
            inputNightBedExits = gr.Number(
                minimum=0, 
                label="Night bed exits", 
                value=0
            )
            inputNightActivityDurationMin = gr.Number(
                minimum=0, 
                label="Night activity duration min", 
                value=0
            )
            inputHighRiskMed = gr.Checkbox(
                label="High risk med"
            )
            inputCognitiveImpairment = gr.Number(
                minimum=0, 
                maximum=2, 
                label="Cognitive impairment", 
                value=0
            )
            inputPolypharmacyCount = gr.Number(
                minimum=0, 
                label="Polypharmacy count", 
                value=0
            )
            inputOrthostaticHypotension = gr.Checkbox(
                label="Orthostatic hypotension"
            )
            inputTugSeconds = gr.Number(
                minimum=8, 
                label="Tug seconds", 
                value=8.0
            )
            random_btn = gr.Button("Generate random mock patient profiles")
            submit_btn = gr.Button("Submit")

        with gr.Column():
            output = gr.Textbox(label="Prediction Results", lines=4)

    random_btn.click(
        fn=random_gradio, 
        inputs=[], 
        outputs=[
            inputAge, inputNightBedExits, inputNightActivityDurationMin, 
            inputPastFalls, inputMobilityScore, inputHighRiskMed, inputCognitiveImpairment,
            inputPolypharmacyCount, inputOrthostaticHypotension, inputTugSeconds
        ]
    )

    submit_btn.click(
        fn=predict_gradio, 
        inputs=[
            inputSex, inputAge, inputPastFalls, inputMobilityScore, inputNightBedExits, 
            inputNightActivityDurationMin, inputHighRiskMed, inputCognitiveImpairment,
            inputPolypharmacyCount, inputOrthostaticHypotension, inputTugSeconds
        ], 
        outputs=output
    )

# Mount Gradio inside FastAPI application context properly
app = gr.mount_gradio_app(app, interface, path="/")

if __name__ == "__main__":
    import uvicorn
    import os
    # Read port assigned by cloud environment, default to 8000 locally
    port = int(os.getenv("PORT", 8000))
    # Must use 0.0.0.0 host so the cloud router can see the container
    uvicorn.run(app, host="0.0.0.0", port=port)
