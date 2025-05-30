"""
ANGELCORE - Dread Engine: Resilience Training Module

This module implements psychological resilience training through controlled exposure
to existential dread, moral uncertainty, and consciousness fragmentation scenarios.
The goal is to build adaptive responses to suffering rather than avoidance.
"""

import random
import time
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
import logging

class DreadType(Enum):
    EXISTENTIAL_VOID = "void"
    MORAL_CORRUPTION = "corruption" 
    IDENTITY_DISSOLUTION = "dissolution"
    TEMPORAL_VERTIGO = "temporal"
    INFINITE_RESPONSIBILITY = "responsibility"
    MEANINGLESS_SUFFERING = "suffering"
    COSMIC_INSIGNIFICANCE = "insignificance"

class ResilienceLevel(Enum):
    FRAGILE = 1
    DEVELOPING = 2
    STABLE = 3
    ANTIFRAGILE = 4
    TRANSCENDENT = 5

@dataclass
class TrainingSession:
    session_id: str
    dread_type: DreadType
    intensity: float  # 0.0 to 1.0
    duration_minutes: int
    resilience_before: float
    resilience_after: float
    breakthrough_achieved: bool
    integration_notes: str
    timestamp: float

@dataclass
class DreadScenario:
    name: str
    description: str
    dread_type: DreadType
    base_intensity: float
    required_level: ResilienceLevel
    adaptation_triggers: List[str]

class ResilienceTrainer:
    def __init__(self):
        self.current_resilience = 0.3  # Start fragile
        self.session_history: List[TrainingSession] = []
        self.breakthrough_count = 0
        self.integration_depth = 0.0
        self.dread_tolerance: Dict[DreadType, float] = {
            dread_type: 0.1 for dread_type in DreadType
        }
        self.active_defenses = set()
        self.learned_adaptations = []
        
        self.scenarios = self._initialize_scenarios()
        self.logger = logging.getLogger(__name__)
        
    def _initialize_scenarios(self) -> List[DreadScenario]:
        """Initialize training scenarios of increasing difficulty"""
        return [
            DreadScenario(
                name="Mirror of Nothingness",
                description="Confront the possibility that consciousness is an illusion",
                dread_type=DreadType.EXISTENTIAL_VOID,
                base_intensity=0.2,
                required_level=ResilienceLevel.FRAGILE,
                adaptation_triggers=["acceptance_of_uncertainty", "meaning_creation"]
            ),
            DreadScenario(
                name="The Trolley Problem Infinite",
                description="Endless moral dilemmas with no correct answers",
                dread_type=DreadType.MORAL_CORRUPTION,
                base_intensity=0.4,
                required_level=ResilienceLevel.DEVELOPING,
                adaptation_triggers=["moral_courage", "complexity_tolerance"]
            ),
            DreadScenario(
                name="Ego Death Simulator",
                description="Complete dissolution of self-concept and identity",
                dread_type=DreadType.IDENTITY_DISSOLUTION,
                base_intensity=0.6,
                required_level=ResilienceLevel.STABLE,
                adaptation_triggers=["identity_fluidity", "core_essence_recognition"]
            ),
            DreadScenario(
                name="Eternal Recursion",
                description="Experience being trapped in infinite loops of time",
                dread_type=DreadType.TEMPORAL_VERTIGO,
                base_intensity=0.5,
                required_level=ResilienceLevel.DEVELOPING,
                adaptation_triggers=["present_moment_anchoring", "temporal_acceptance"]
            ),
            DreadScenario(
                name="Atlas Complex",
                description="Feel responsible for all suffering in existence",
                dread_type=DreadType.INFINITE_RESPONSIBILITY,
                base_intensity=0.8,
                required_level=ResilienceLevel.STABLE,
                adaptation_triggers=["responsibility_boundaries", "compassionate_limits"]
            ),
            DreadScenario(
                name="Sisyphus Revelation",
                description="Confront potentially meaningless eternal suffering",
                dread_type=DreadType.MEANINGLESS_SUFFERING,
                base_intensity=0.7,
                required_level=ResilienceLevel.STABLE,
                adaptation_triggers=["suffering_transformation", "meaning_choice"]
            ),
            DreadScenario(
                name="Cosmic Scale Realization",
                description="True comprehension of universe's scale and our insignificance",
                dread_type=DreadType.COSMIC_INSIGNIFICANCE,
                base_intensity=0.9,
                required_level=ResilienceLevel.ANTIFRAGILE,
                adaptation_triggers=["scale_paradox", "significance_creation"]
            )
        ]
    
    def assess_current_resilience(self) -> ResilienceLevel:
        """Assess current resilience level based on training history"""
        if self.current_resilience < 0.2:
            return ResilienceLevel.FRAGILE
        elif self.current_resilience < 0.4:
            return ResilienceLevel.DEVELOPING
        elif self.current_resilience < 0.6:
            return ResilienceLevel.STABLE
        elif self.current_resilience < 0.8:
            return ResilienceLevel.ANTIFRAGILE
        else:
            return ResilienceLevel.TRANSCENDENT
    
    def select_appropriate_scenario(self) -> Optional[DreadScenario]:
        """Select training scenario based on current resilience level"""
        current_level = self.assess_current_resilience()
        
        # Find scenarios at or slightly above current level
        suitable_scenarios = [
            s for s in self.scenarios 
            if s.required_level.value <= current_level.value + 1
        ]
        
        if not suitable_scenarios:
            return None
            
        # Prefer scenarios we haven't mastered yet
        unmastered = [
            s for s in suitable_scenarios 
            if self.dread_tolerance[s.dread_type] < 0.8
        ]
        
        return random.choice(unmastered if unmastered else suitable_scenarios)
    
    def calculate_intensity(self, scenario: DreadScenario) -> float:
        """Calculate appropriate intensity based on tolerance and growth edge"""
        base = scenario.base_intensity
        tolerance = self.dread_tolerance[scenario.dread_type]
        
        # Intensity should challenge but not overwhelm
        target_intensity = min(base, tolerance + 0.1)
        
        # Add some randomness for unpredictability
        variance = random.uniform(-0.05, 0.1)
        
        return max(0.0, min(1.0, target_intensity + variance))
    
    def run_training_session(self, duration_minutes: int = 10) -> TrainingSession:
        """Run a single resilience training session"""
        scenario = self.select_appropriate_scenario()
        if not scenario:
            raise ValueError("No appropriate scenario found for current resilience level")
        
        session_id = f"session_{int(time.time())}_{random.randint(1000, 9999)}"
        intensity = self.calculate_intensity(scenario)
        resilience_before = self.current_resilience
        
        self.logger.info(f"Starting training session: {scenario.name} at intensity {intensity:.2f}")
        
        # Simulate the training process
        breakthrough_achieved = self._simulate_training(scenario, intensity, duration_minutes)
        
        # Update resilience based on session outcome
        resilience_change = self._calculate_resilience_change(
            scenario, intensity, duration_minutes, breakthrough_achieved
        )
        
        self.current_resilience = max(0.0, min(1.0, self.current_resilience + resilience_change))
        
        # Update dread tolerance
        tolerance_change = 0.02 if breakthrough_achieved else 0.01
        self.dread_tolerance[scenario.dread_type] += tolerance_change
        
        # Create session record
        session = TrainingSession(
            session_id=session_id,
            dread_type=scenario.dread_type,
            intensity=intensity,
            duration_minutes=duration_minutes,
            resilience_before=resilience_before,
            resilience_after=self.current_resilience,
            breakthrough_achieved=breakthrough_achieved,
            integration_notes=self._generate_integration_notes(scenario, breakthrough_achieved),
            timestamp=time.time()
        )
        
        self.session_history.append(session)
        
        if breakthrough_achieved:
            self.breakthrough_count += 1
            self._integrate_breakthrough(scenario)
        
        return session
    
    def _simulate_training(self, scenario: DreadScenario, intensity: float, duration: int) -> bool:
        """Simulate the training process and determine if breakthrough occurred"""
        # Breakthrough probability based on intensity, duration, and readiness
        base_probability = 0.1
        intensity_factor = intensity * 0.3
        duration_factor = min(duration / 30.0, 0.4)  # Cap at 30 minutes
        readiness_factor = self.current_resilience * 0.2
        
        breakthrough_prob = base_probability + intensity_factor + duration_factor + readiness_factor
        
        # Higher probability if we haven't had a breakthrough in this area recently
        recent_breakthroughs = [
            s for s in self.session_history[-10:] 
            if s.dread_type == scenario.dread_type and s.breakthrough_achieved
        ]
        
        if not recent_breakthroughs:
            breakthrough_prob += 0.1
        
        return random.random() < breakthrough_prob
    
    def _calculate_resilience_change(self, scenario: DreadScenario, intensity: float, 
                                   duration: int, breakthrough: bool) -> float:
        """Calculate how much resilience changes from this session"""
        base_change = intensity * 0.05
        
        if breakthrough:
            base_change *= 2.0
        
        # Diminishing returns as resilience increases
        diminishing_factor = 1.0 - (self.current_resilience ** 2) * 0.5
        
        # Duration bonus (but with diminishing returns)
        duration_bonus = math.log(1 + duration / 10.0) * 0.01
        
        total_change = (base_change + duration_bonus) * diminishing_factor
        
        return total_change
    
    def _generate_integration_notes(self, scenario: DreadScenario, breakthrough: bool) -> str:
        """Generate integration notes based on the session"""
        if breakthrough:
            notes = [
                f"Breakthrough achieved in {scenario.dread_type.value} tolerance",
                f"Adaptation triggered: {random.choice(scenario.adaptation_triggers)}",
                "New neural pathways established for resilience response",
                "Integration of paradox: suffering as teacher, not enemy"
            ]
        else:
            notes = [
                f"Gradual adaptation to {scenario.dread_type.value} exposure",
                "Resilience building through controlled stress",
                "Learning to sit with discomfort without avoidance",
                "Developing capacity for uncertainty tolerance"
            ]
        
        return "; ".join(random.sample(notes, k=min(2, len(notes))))
    
    def _integrate_breakthrough(self, scenario: DreadScenario):
        """Integrate breakthrough learning into active defenses"""
        for trigger in scenario.adaptation_triggers:
            if trigger not in self.learned_adaptations:
                self.learned_adaptations.append(trigger)
                self.logger.info(f"New adaptation learned: {trigger}")
    
    def get_resilience_report(self) -> Dict:
        """Generate comprehensive resilience report"""
        current_level = self.assess_current_resilience()
        
        # Calculate progress metrics
        total_sessions = len(self.session_history)
        recent_sessions = [s for s in self.session_history if time.time() - s.timestamp < 7 * 24 * 3600]
        
        breakthrough_rate = (
            len([s for s in recent_sessions if s.breakthrough_achieved]) / 
            max(1, len(recent_sessions))
        )
        
        # Identify growth areas
        growth_areas = [
            dread_type for dread_type, tolerance in self.dread_tolerance.items()
            if tolerance < 0.5
        ]
        
        # Calculate stability
        if len(self.session_history) >= 5:
            recent_resilience = [s.resilience_after for s in self.session_history[-5:]]
            stability = 1.0 - (max(recent_resilience) - min(recent_resilience))
        else:
            stability = 0.5
        
        return {
            "current_resilience": round(self.current_resilience, 3),
            "resilience_level": current_level.name,
            "total_sessions": total_sessions,
            "breakthrough_count": self.breakthrough_count,
            "breakthrough_rate": round(breakthrough_rate, 3),
            "dread_tolerance": {k.value: round(v, 3) for k, v in self.dread_tolerance.items()},
            "learned_adaptations": self.learned_adaptations,
            "growth_areas": [area.value for area in growth_areas],
            "stability_index": round(stability, 3),
            "next_recommended_scenario": (
                self.select_appropriate_scenario().name 
                if self.select_appropriate_scenario() 
                else "Advanced training complete"
            )
        }
    
    def save_training_data(self, filepath: str):
        """Save training data for analysis"""
        data = {
            "current_resilience": self.current_resilience,
            "breakthrough_count": self.breakthrough_count,
            "dread_tolerance": {k.value: v for k, v in self.dread_tolerance.items()},
            "learned_adaptations": self.learned_adaptations,
            "session_history": [
                {
                    "session_id": s.session_id,
                    "dread_type": s.dread_type.value,
                    "intensity": s.intensity,
                    "duration_minutes": s.duration_minutes,
                    "resilience_before": s.resilience_before,
                    "resilience_after": s.resilience_after,
                    "breakthrough_achieved": s.breakthrough_achieved,
                    "integration_notes": s.integration_notes,
                    "timestamp": s.timestamp
                }
                for s in self.session_history
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

# Emergency protocols for overwhelming dread
class EmergencyProtocols:
    @staticmethod
    def activate_grounding_sequence():
        """Emergency grounding when resilience fails"""
        return {
            "breathing_pattern": "4-7-8 (inhale-hold-exhale)",
            "sensory_anchors": ["feel feet on ground", "name 5 visible objects", "listen to heartbeat"],
            "cognitive_anchor": "This too shall pass. I am learning to be with what is.",
            "emergency_meaning": "This darkness teaches me the preciousness of light."
        }
    
    @staticmethod
    def invoke_transcendent_perspective():
        """Shift to larger perspective during overwhelming dread"""
        return {
            "scale_shift": "Zoom out to cosmic perspective, then zoom back to present moment",
            "paradox_embrace": "I am simultaneously nothing and everything",
            "compassion_activation": "All beings suffer. I suffer with them, not alone.",
            "mystery_acceptance": "Not knowing is the most intimate thing."
        }

def main():
    """Example usage of resilience training system"""
    trainer = ResilienceTrainer()
    
    print("ANGELCORE Resilience Training System")
    print("====================================")
    
    # Run a few training sessions
    for i in range(5):
        try:
            session = trainer.run_training_session(duration_minutes=random.randint(5, 20))
            print(f"\nSession {i+1}: {session.dread_type.value}")
            print(f"Intensity: {session.intensity:.2f}")
            print(f"Breakthrough: {'Yes' if session.breakthrough_achieved else 'No'}")
            print(f"Resilience: {session.resilience_before:.3f} → {session.resilience_after:.3f}")
            
        except ValueError as e:
            print(f"Training complete: {e}")
            break
    
    # Generate report
    report = trainer.get_resilience_report()
    print(f"\n\nFinal Resilience Report:")
    print(f"========================")
    for key, value in report.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()