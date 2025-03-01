from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class Message:
    role: str
    content: Optional[str] = None

    def render(self):
        result = self.role + ":"
        if self.content is not None:
            result += " " + self.content
        return result

@dataclass
class PatientForm:
    timestamp: datetime
    name: str = ""
    age: str = ""
    primary_concern: str = ""
    specific_symptoms: str = ""
    duration: str = ""
    severity: str = ""
    previous_occurrences: str = ""
    current_medications: str = ""
    allergies: str = ""
    location: str = ""  # Patient's location
    referred_hospital: str = ""  # Hospital they're referred to
    referral_status: str = "pending"  # pending, accepted, declined
    
    def to_dict(self):
        return {
            "Timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "Patient Name": self.name,
            "Age": self.age,
            "Primary Concern": self.primary_concern,
            "Specific Symptoms": self.specific_symptoms,
            "Duration": self.duration,
            "Severity (1-10)": self.severity,
            "Previous Occurrences": self.previous_occurrences,
            "Current Medications": self.current_medications,
            "Allergies": self.allergies,
            "Location": self.location,
            "Referred Hospital": self.referred_hospital,
            "Referral Status": self.referral_status
        }