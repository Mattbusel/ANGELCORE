"""
ANGELCORE - Dread Engine: Collapse Index Module

This module monitors and predicts systemic collapse across multiple dimensions
of consciousness, reality perception, and existential coherence. It tracks
the precarious balance between order and chaos in conscious systems.
"""

import time
import math
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json
import warnings

class CollapseType(Enum):
    COGNITIVE_DISSONANCE = "cognitive"
    REALITY_BREAKDOWN = "reality" 
    TEMPORAL_FRACTURE = "temporal"
    IDENTITY_DISSOLUTION = "identity"
    MORAL_VERTIGO = "moral"
    EXISTENTIAL_VOID = "existential"
    SYSTEM_OVERLOAD = "overload"
    MEANING_COLLAPSE = "meaning"
    RELATIONAL_BREAKDOWN = "relational"
    DIMENSIONAL_BLEED = "dimensional"

class SeverityLevel(Enum):
    STABLE = 0
    TREMOR = 1
    FRACTURE = 2
    CRITICAL = 3
    COLLAPSE = 4
    VOID = 5

@dataclass
class CollapseIndicator:
    name: str
    collapse_type: CollapseType
    current_value: float
    threshold: float
    rate_of_change: float
    instability_factor: float
    last_updated: float
    historical_values: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update(self, new_value: float):
        """Update indicator with new measurement"""
        old_value = self.current_value
        self.current_value = new_value
        self.rate_of_change = new_value - old_value
        self.last_updated = time.time()
        self.historical_values.append(new_value)
        
        # Calculate instability based on recent volatility
        if len(self.historical_values) > 10:
            recent_values = list(self.historical_values)[-10:]
            self.instability_factor = np.std(recent_values) / (np.mean(recent_values) + 0.001)

@dataclass
class CollapseEvent:
    event_id: str
    timestamp: float
    collapse_type: CollapseType
    severity: SeverityLevel
    triggering_indicators: List[str]
    cascade_effects: List[str]
    recovery_time: Optional[float]
    integration_achieved: bool
    notes: str

class CollapseIndex:
    def __init__(self):
        self.indicators: Dict[str, CollapseIndicator] = {}
        self.collapse_history: List[CollapseEvent] = []
        self.cascade_multipliers: Dict[CollapseType, Dict[CollapseType, float]] = {}
        self.recovery_factors: Dict[CollapseType, float] = {}
        self.dimensional_bleed_rate = 0.0
        self.void_proximity = 0.0
        self.system_coherence = 1.0
        self.emergency_threshold = 0.8
        
        self._initialize_indicators()
        self._initialize_cascade_matrix()
        self._initialize_recovery_factors()
        
    def _initialize_indicators(self):
        """Initialize all collapse indicators with baseline values"""
        
        # Cognitive Dissonance Indicators
        self.indicators["belief_conflict"] = CollapseIndicator(
            "Conflicting Belief Systems",
            CollapseType.COGNITIVE_DISSONANCE,
            0.2, 0.7, 0.0, 0.0, time.time()
        )
        
        self.indicators["logic_inconsistency"] = CollapseIndicator(
            "Logical Inconsistency Load",
            CollapseType.COGNITIVE_DISSONANCE,
            0.1, 0.6, 0.0, 0.0, time.time()
        )
        
        # Reality Breakdown Indicators
        self.indicators["perception_drift"] = CollapseIndicator(
            "Perceptual Drift from Consensus",
            CollapseType.REALITY_BREAKDOWN,
            0.15, 0.8, 0.0, 0.0, time.time()
        )
        
        self.indicators["hallucination_frequency"] = CollapseIndicator(
            "Hallucination Event Frequency",
            CollapseType.REALITY_BREAKDOWN,
            0.05, 0.4, 0.0, 0.0, time.time()
        )
        
        self.indicators["consensus_deviation"] = CollapseIndicator(
            "Deviation from Consensus Reality",
            CollapseType.REALITY_BREAKDOWN,
            0.1, 0.6, 0.0, 0.0, time.time()
        )
        
        # Temporal Fracture Indicators
        self.indicators["time_perception_drift"] = CollapseIndicator(
            "Temporal Perception Anomalies",
            CollapseType.TEMPORAL_FRACTURE,
            0.1, 0.7, 0.0, 0.0, time.time()
        )
        
        self.indicators["causal_loop_density"] = CollapseIndicator(
            "Causal Loop Formation Rate",
            CollapseType.TEMPORAL_FRACTURE,
            0.05, 0.5, 0.0, 0.0, time.time()
        )
        
        # Identity Dissolution Indicators
        self.indicators["self_boundary_coherence"] = CollapseIndicator(
            "Self-Other Boundary Integrity",
            CollapseType.IDENTITY_DISSOLUTION,
            0.8, 0.3, 0.0, 0.0, time.time()  # Inverted - high is good
        )
        
        self.indicators["narrative_continuity"] = CollapseIndicator(
            "Personal Narrative Continuity",
            CollapseType.IDENTITY_DISSOLUTION,
            0.7, 0.2, 0.0, 0.0, time.time()  # Inverted
        )
        
        # Moral Vertigo Indicators
        self.indicators["ethical_uncertainty"] = CollapseIndicator(
            "Ethical Decision Paralysis",
            CollapseType.MORAL_VERTIGO,
            0.3, 0.8, 0.0, 0.0, time.time()
        )
        
        self.indicators["value_system_conflict"] = CollapseIndicator(
            "Core Value System Conflicts",
            CollapseType.MORAL_VERTIGO,
            0.2, 0.7, 0.0, 0.0, time.time()
        )
        
        # Existential Void Indicators
        self.indicators["meaning_vacuum"] = CollapseIndicator(
            "Meaning Vacuum Intensity",
            CollapseType.EXISTENTIAL_VOID,
            0.4, 0.8, 0.0, 0.0, time.time()
        )
        
        self.indicators["purpose_dissolution"] = CollapseIndicator(
            "Purpose Structure Dissolution",
            CollapseType.EXISTENTIAL_VOID,
            0.3, 0.7, 0.0, 0.0, time.time()
        )
        
        # System Overload Indicators
        self.indicators["processing_saturation"] = CollapseIndicator(
            "Cognitive Processing Saturation",
            CollapseType.SYSTEM_OVERLOAD,
            0.4, 0.9, 0.0, 0.0, time.time()
        )
        
        self.indicators["paradox_accumulation"] = CollapseIndicator(
            "Unresolved Paradox Accumulation",
            CollapseType.SYSTEM_OVERLOAD,
            0.2, 0.6, 0.0, 0.0, time.time()
        )
        
        # Relational Breakdown Indicators
        self.indicators["empathy_failure_rate"] = CollapseIndicator(
            "Empathic Connection Failure Rate",
            CollapseType.RELATIONAL_BREAKDOWN,
            0.1, 0.5, 0.0, 0.0, time.time()
        )
        
        self.indicators["trust_system_decay"] = CollapseIndicator(
            "Trust System Degradation",
            CollapseType.RELATIONAL_BREAKDOWN,
            0.2, 0.6, 0.0, 0.0, time.time()
        )
        
        # Dimensional Bleed Indicators
        self.indicators["reality_layer_permeability"] = CollapseIndicator(
            "Reality Layer Boundary Permeability",
            CollapseType.DIMENSIONAL_BLEED,
            0.1, 0.4, 0.0, 0.0, time.time()
        )
    
    def _initialize_cascade_matrix(self):
        """Initialize cascade effect multipliers between collapse types"""
        self.cascade_multipliers = {
            CollapseType.COGNITIVE_DISSONANCE: {
                CollapseType.REALITY_BREAKDOWN: 0.4,
                CollapseType.MORAL_VERTIGO: 0.6,
                CollapseType.SYSTEM_OVERLOAD: 0.5
            },
            CollapseType.REALITY_BREAKDOWN: {
                CollapseType.IDENTITY_DISSOLUTION: 0.7,
                CollapseType.TEMPORAL_FRACTURE: 0.5,
                CollapseType.EXISTENTIAL_VOID: 0.3,
                CollapseType.DIMENSIONAL_BLEED: 0.8
            },
            CollapseType.TEMPORAL_FRACTURE: {
                CollapseType.REALITY_BREAKDOWN: 0.6,
                CollapseType.IDENTITY_DISSOLUTION: 0.4,
                CollapseType.COGNITIVE_DISSONANCE: 0.3
            },
            CollapseType.IDENTITY_DISSOLUTION: {
                CollapseType.EXISTENTIAL_VOID: 0.9,
                CollapseType.RELATIONAL_BREAKDOWN: 0.7,
                CollapseType.TEMPORAL_FRACTURE: 0.3
            },
            CollapseType.MORAL_VERTIGO: {
                CollapseType.EXISTENTIAL_VOID: 0.5,
                CollapseType.IDENTITY_DISSOLUTION: 0.4,
                CollapseType.RELATIONAL_BREAKDOWN: 0.6
            },
            CollapseType.EXISTENTIAL_VOID: {
                CollapseType.MEANING_COLLAPSE: 0.9,
                CollapseType.SYSTEM_OVERLOAD: 0.4,
                CollapseType.DIMENSIONAL_BLEED: 0.2
            },
            CollapseType.SYSTEM_OVERLOAD: {
                CollapseType.COGNITIVE_DISSONANCE: 0.8,
                CollapseType.REALITY_BREAKDOWN: 0.6,
                CollapseType.TEMPORAL_FRACTURE: 0.4
            },
            CollapseType.RELATIONAL_BREAKDOWN: {
                CollapseType.IDENTITY_DISSOLUTION: 0.5,
                CollapseType.EXISTENTIAL_VOID: 0.3,
                CollapseType.MORAL_VERTIGO: 0.4
            },
            CollapseType.DIMENSIONAL_BLEED: {
                CollapseType.REALITY_BREAKDOWN: 0.9,
                CollapseType.TEMPORAL_FRACTURE: 0.7,
                CollapseType.SYSTEM_OVERLOAD: 0.5
            }
        }
    
    def _initialize_recovery_factors(self):
        """Initialize recovery factors for different collapse types"""
        self.recovery_factors = {
            CollapseType.COGNITIVE_DISSONANCE: 0.7,
            CollapseType.REALITY_BREAKDOWN: 0.4,
            CollapseType.TEMPORAL_FRACTURE: 0.3,
            CollapseType.IDENTITY_DISSOLUTION: 0.5,
            CollapseType.MORAL_VERTIGO: 0.6,
            CollapseType.EXISTENTIAL_VOID: 0.2,
            CollapseType.SYSTEM_OVERLOAD: 0.8,
            CollapseType.MEANING_COLLAPSE: 0.1,
            CollapseType.RELATIONAL_BREAKDOWN: 0.6,
            CollapseType.DIMENSIONAL_BLEED: 0.1
        }
    
    def update_indicator(self, indicator_name: str, value: float):
        """Update a specific collapse indicator"""
        if indicator_name in self.indicators:
            self.indicators[indicator_name].update(value)
            self._check_cascade_effects(indicator_name)
            self._update_system_coherence()
        else:
            raise ValueError(f"Unknown indicator: {indicator_name}")
    
    def _check_cascade_effects(self, primary_indicator: str):
        """Check if indicator breach triggers cascade effects"""
        indicator = self.indicators[primary_indicator]
        
        if indicator.current_value > indicator.threshold:
            collapse_type = indicator.collapse_type
            
            # Apply cascade effects to related systems
            if collapse_type in self.cascade_multipliers:
                for target_type, multiplier in self.cascade_multipliers[collapse_type].items():
                    self._apply_cascade_effect(target_type, multiplier * 0.1)
    
    def _apply_cascade_effect(self, target_type: CollapseType, effect_strength: float):
        """Apply cascade effect to indicators of target type"""
        for indicator in self.indicators.values():
            if indicator.collapse_type == target_type:
                cascade_increase = effect_strength * (1 + indicator.instability_factor)
                new_value = min(1.0, indicator.current_value + cascade_increase)
                indicator.update(new_value)
    
    def _update_system_coherence(self):
        """Update overall system coherence based on all indicators"""
        total_stress = 0.0
        critical_breaches = 0
        
        for indicator in self.indicators.values():
            if indicator.current_value > indicator.threshold:
                breach_severity = indicator.current_value - indicator.threshold
                total_stress += breach_severity * (1 + indicator.instability_factor)
                critical_breaches += 1
        
        # System coherence decreases with stress and critical breaches
        coherence_loss = (total_stress * 0.5) + (critical_breaches * 0.1)
        self.system_coherence = max(0.0, 1.0 - coherence_loss)
        
        # Update void proximity (approaches 1.0 as coherence collapses)
        self.void_proximity = 1.0 - self.system_coherence
        
        # Update dimensional bleed rate
        self.dimensional_bleed_rate = min(0.5, self.void_proximity * 0.3)
    
    def get_collapse_severity(self) -> SeverityLevel:
        """Calculate overall collapse severity"""
        if self.system_coherence > 0.8:
            return SeverityLevel.STABLE
        elif self.system_coherence > 0.6:
            return SeverityLevel.TREMOR
        elif self.system_coherence > 0.4:
            return SeverityLevel.FRACTURE
        elif self.system_coherence > 0.2:
            return SeverityLevel.CRITICAL
        elif self.system_coherence > 0.05:
            return SeverityLevel.COLLAPSE
        else:
            return SeverityLevel.VOID
    
    def predict_collapse_probability(self, time_horizon_hours: float = 24.0) -> Dict[CollapseType, float]:
        """Predict probability of collapse by type within time horizon"""
        predictions = {}
        
        for collapse_type in CollapseType:
            # Get indicators of this type
            type_indicators = [
                ind for ind in self.indicators.values() 
                if ind.collapse_type == collapse_type
            ]
            
            if not type_indicators:
                predictions[collapse_type] = 0.0
                continue
            
            # Calculate probability based on current values, rates, and instability
            total_risk = 0.0
            for indicator in type_indicators:
                # Distance from threshold
                threshold_distance = max(0, indicator.threshold - indicator.current_value)
                
                # Rate of approach to threshold
                if indicator.rate_of_change > 0:
                    time_to_threshold = threshold_distance / indicator.rate_of_change
                    urgency_factor = max(0, 1.0 - (time_to_threshold / time_horizon_hours))
                else:
                    urgency_factor = 0.0
                
                # Base risk from current proximity to threshold
                proximity_risk = 1.0 - (threshold_distance / indicator.threshold)
                
                # Instability increases unpredictability
                instability_risk = indicator.instability_factor * 0.5
                
                indicator_risk = min(1.0, proximity_risk + urgency_factor + instability_risk)
                total_risk += indicator_risk
            
            # Average risk across indicators, with cascade effects
            avg_risk = total_risk / len(type_indicators)
            
            # Add cascade amplification from other collapsing systems
            cascade_amplification = 0.0
            for other_type, multipliers in self.cascade_multipliers.items():
                if collapse_type in multipliers:
                    other_risk = self._get_type_current_risk(other_type)
                    cascade_amplification += other_risk * multipliers[collapse_type] * 0.2
            
            final_probability = min(0.95, avg_risk + cascade_amplification)
            predictions[collapse_type] = final_probability
        
        return predictions
    
    def _get_type_current_risk(self, collapse_type: CollapseType) -> float:
        """Get current risk level for a collapse type"""
        type_indicators = [
            ind for ind in self.indicators.values() 
            if ind.collapse_type == collapse_type
        ]
        
        if not type_indicators:
            return 0.0
        
        total_risk = sum(
            max(0, ind.current_value - ind.threshold) / (1.0 - ind.threshold)
            for ind in type_indicators
        )
        
        return min(1.0, total_risk / len(type_indicators))
    
    def simulate_collapse_event(self, collapse_type: CollapseType, severity: float = 0.8):
        """Simulate a collapse event and its effects"""
        event_id = f"collapse_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Determine severity level
        if severity < 0.2:
            sev_level = SeverityLevel.TREMOR
        elif severity < 0.4:
            sev_level = SeverityLevel.FRACTURE
        elif severity < 0.6:
            sev_level = SeverityLevel.CRITICAL
        elif severity < 0.8:
            sev_level = SeverityLevel.COLLAPSE
        else:
            sev_level = SeverityLevel.VOID
        
        # Identify triggering indicators
        triggering_indicators = [
            name for name, ind in self.indicators.items()
            if ind.collapse_type == collapse_type and ind.current_value > ind.threshold
        ]
        
        # Apply immediate effects to system
        direct_impact = severity * 0.3
        self.system_coherence = max(0.0, self.system_coherence - direct_impact)
        
        # Generate cascade effects
        cascade_effects = []
        if collapse_type in self.cascade_multipliers:
            for target_type, multiplier in self.cascade_multipliers[collapse_type].items():
                cascade_strength = severity * multiplier * 0.4
                self._apply_cascade_effect(target_type, cascade_strength)
                cascade_effects.append(f"{target_type.value} +{cascade_strength:.2f}")
        
        # Estimate recovery time
        base_recovery = self.recovery_factors.get(collapse_type, 0.5)
        recovery_time = (severity / base_recovery) * 24.0  # Hours
        
        # Create collapse event record
        event = CollapseEvent(
            event_id=event_id,
            timestamp=time.time(),
            collapse_type=collapse_type,
            severity=sev_level,
            triggering_indicators=triggering_indicators,
            cascade_effects=cascade_effects,
            recovery_time=recovery_time,
            integration_achieved=random.random() < base_recovery,
            notes=f"Simulated {collapse_type.value} collapse at severity {severity:.2f}"
        )
        
        self.collapse_history.append(event)
        return event
    
    def get_vulnerability_analysis(self) -> Dict:
        """Analyze current system vulnerabilities"""
        vulnerabilities = {}
        
        for collapse_type in CollapseType:
            type_indicators = [
                ind for ind in self.indicators.values() 
                if ind.collapse_type == collapse_type
            ]
            
            if not type_indicators:
                continue
            
            # Calculate vulnerability metrics
            avg_proximity = np.mean([
                ind.current_value / ind.threshold for ind in type_indicators
            ])
            
            max_instability = max([ind.instability_factor for ind in type_indicators])
            
            worst_indicator = max(type_indicators, key=lambda x: x.current_value / x.threshold)
            
            vulnerabilities[collapse_type.value] = {
                "proximity_to_collapse": round(avg_proximity, 3),
                "instability_factor": round(max_instability, 3),
                "most_vulnerable_indicator": worst_indicator.name,
                "estimated_time_to_collapse": self._estimate_time_to_collapse(type_indicators),
                "cascade_potential": len(self.cascade_multipliers.get(collapse_type, {}))
            }
        
        return vulnerabilities
    
    def _estimate_time_to_collapse(self, indicators: List[CollapseIndicator]) -> Optional[float]:
        """Estimate time until collapse for a set of indicators"""
        min_time = float('inf')
        
        for indicator in indicators:
            if indicator.rate_of_change <= 0:
                continue
            
            remaining_distance = indicator.threshold - indicator.current_value
            if remaining_distance <= 0:
                return 0.0
            
            time_to_threshold = remaining_distance / indicator.rate_of_change
            min_time = min(min_time, time_to_threshold)
        
        return min_time if min_time != float('inf') else None
    
    def emergency_stabilization_protocol(self):
        """Emergency protocol when system approaches critical collapse"""
        if self.system_coherence < 0.3:
            # Emergency damping of all indicators
            for indicator in self.indicators.values():
                damping_factor = 0.1 * (1.0 - self.system_coherence)
                new_value = indicator.current_value * (1.0 - damping_factor)
                indicator.update(new_value)
            
            # Force coherence boost
            self.system_coherence = min(1.0, self.system_coherence + 0.2)
            
            return {
                "protocol_activated": True,
                "damping_applied": True,
                "coherence_boost": 0.2,
                "message": "Emergency stabilization protocols engaged"
            }
        
        return {"protocol_activated": False}
    
    def get_status_report(self) -> Dict:
        """Get comprehensive collapse index status report"""
        severity = self.get_collapse_severity()
        predictions = self.predict_collapse_probability()
        vulnerabilities = self.get_vulnerability_analysis()
        
        # Find most critical indicators
        critical_indicators = [
            (name, ind) for name, ind in self.indicators.items()
            if ind.current_value > ind.threshold
        ]
        
        critical_indicators.sort(key=lambda x: x[1].current_value / x[1].threshold, reverse=True)
        
        return {
            "system_coherence": round(self.system_coherence, 3),
            "void_proximity": round(self.void_proximity, 3),
            "dimensional_bleed_rate": round(self.dimensional_bleed_rate, 3),
            "overall_severity": severity.name,
            "critical_indicators": [
                {
                    "name": name,
                    "type": ind.collapse_type.value,
                    "current_value": round(ind.current_value, 3),
                    "threshold": ind.threshold,
                    "breach_severity": round((ind.current_value - ind.threshold) / (1.0 - ind.threshold), 3)
                }
                for name, ind in critical_indicators[:5]
            ],
            "collapse_predictions_24h": {
                k.value: round(v, 3) for k, v in predictions.items() if v > 0.1
            },
            "vulnerability_analysis": vulnerabilities,
            "recent_collapses": len([
                e for e in self.collapse_history
                if time.time() - e.timestamp < 24 * 3600
            ]),
            "emergency_threshold_breached": self.system_coherence < self.emergency_threshold
        }

def main():
    """Demonstration of collapse index system"""
    print("ANGELCORE Collapse Index System")
    print("===============================")
    
    collapse_index = CollapseIndex()
    
    # Simulate some stress on the system
    print("\nSimulating system stress...")
    
    # Gradual increase in cognitive dissonance
    for i in range(10):
        belief_conflict = 0.2 + (i * 0.08)
        collapse_index.update_indicator("belief_conflict", belief_conflict)
        
        # Some reality breakdown
        perception_drift = 0.15 + (i * 0.05)
        collapse_index.update_indicator("perception_drift", perception_drift)
        
        # Existential pressure
        meaning_vacuum = 0.4 + (i * 0.04)
        collapse_index.update_indicator("meaning_vacuum", meaning_vacuum)
        
        if i % 3 == 0:
            status = collapse_index.get_status_report()
            print(f"\nStep {i+1}:")
            print(f"System Coherence: {status['system_coherence']}")
            print(f"Severity: {status['overall_severity']}")
            print(f"Void Proximity: {status['void_proximity']}")
    
    # Final status report
    print("\n" + "="*50)
    print("FINAL STATUS REPORT")
    print("="*50)
    
    final_status = collapse_index.get_status_report()
    
    for key, value in final_status.items():
        if key != "vulnerability_analysis":
            print(f"{key}: {value}")
    
    # Check if emergency protocols needed
    emergency_result = collapse_index.emergency_stabilization_protocol()
    if emergency_result["protocol_activated"]:
        print(f"\n🚨 {emergency_result['message']}")

if __name__ == "__main__":
    main()