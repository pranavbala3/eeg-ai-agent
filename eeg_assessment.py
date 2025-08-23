import numpy as np
from typing import Dict, List, Optional
import json
import uuid

from EEGPTPatientAssessment import EEGPTPatientAssessment

def analyze_patient_eeg(voltage_data: np.ndarray, sampling_rate: int = 250, 
                       channel_names: Optional[List[str]] = None) -> Dict[str, any]:
    """
    **MAIN FUNCTION: Analyze patient EEG and return interpretable assessment**
    
    Args:
        voltage_data: Raw EEG voltages in volts, shape (channels, timepoints)
        sampling_rate: Sampling frequency in Hz
        channel_names: Optional channel names
        
    Returns:
        Complete interpretable patient assessment
    """
    
    # Initialize assessment system
    assessor = EEGPTPatientAssessment()
    
    # Analyze EEG
    assessment = assessor.analyze_eeg(voltage_data, sampling_rate, channel_names)
    
    return assessment

def generate_synthetic_eeg_data(n_channels=8, n_timepoints=1000, sampling_rate=250):
    """Generate synthetic EEG voltage data for demonstration or testing purposes."""
    np.random.seed(42)
    t = np.linspace(0, n_timepoints / sampling_rate, n_timepoints)
    voltage_data = np.zeros((n_channels, n_timepoints))
    for i in range(n_channels):
        voltage_data[i] += 50e-6 * np.sin(2 * np.pi * 10 * t)  # Alpha rhythm (10 Hz)
        voltage_data[i] += 20e-6 * np.sin(2 * np.pi * 20 * t)  # Beta activity (20 Hz)
        voltage_data[i] += 10e-6 * np.random.randn(n_timepoints)  # Noise
    channel_names = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
    return voltage_data, sampling_rate, channel_names

def run_patient_assessment(voltage_data, sampling_rate=250, channel_names=None):
    """Run the EEGPT model to assess patient EEG data and return the assessment dictionary."""
    return analyze_patient_eeg(voltage_data, sampling_rate, channel_names)

def format_patient_assessment(assessment):
    """Format the patient assessment dictionary into a readable string report."""
    summary = assessment.get('patient_summary', {})
    interpretations = assessment.get('clinical_interpretations', {})
    recommendations = assessment.get('recommendations', [])
    important_channels = assessment.get('interpretability', {}).get('most_important_channels', [])

    report = []
    report.append("EEGPT Patient Assessment System\n" + "=" * 50)
    report.append("\nPATIENT ASSESSMENT RESULTS:")
    report.append("-" * 30)
    report.append(f"Overall State: {summary.get('overall_brain_state', 'N/A')}")
    report.append(f"Alertness: {summary.get('alertness_level', 'N/A')}")
    report.append(f"Stress Level: {summary.get('stress_level', 'N/A')}")
    report.append(f"Attention: {summary.get('attention_score', 'N/A')}")
    report.append(f"Cognitive State: {summary.get('cognitive_state', 'N/A')}")
    report.append(f"Seizure Risk: {summary.get('seizure_risk', 'N/A')}")
    report.append(f"Confidence: {summary.get('confidence_level', 'N/A')}")

    report.append("\nCLINICAL INTERPRETATIONS:")
    report.append("-" * 30)
    for key, value in interpretations.items():
        report.append(f"{key.title()}: {value}")

    report.append("\nRECOMMENDATIONS:")
    report.append("-" * 30)
    for i, rec in enumerate(recommendations, 1):
        report.append(f"{i}. {rec}")

    report.append("\nMOST IMPORTANT CHANNELS:")
    report.append(f"{', '.join(important_channels)}")

    return '\n'.join(report)

def save_to_json(dictionary_data, filepath):
    """Save a dictionary to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(dictionary_data, f, indent=2)

def main():
    """Main entry point for running EEGPT patient assessment workflow."""
    voltage_data, sampling_rate, channel_names = generate_synthetic_eeg_data()

    assessment = run_patient_assessment(voltage_data, sampling_rate, channel_names)
    report = format_patient_assessment(assessment)
    print(report)

    save_path = f'data/patient_eeg_assessment-{uuid.uuid4()}.json'
    save_to_json(assessment, save_path)
    print("\n✓ Complete assessment saved to 'patient_eeg_assessment.json'")


main()