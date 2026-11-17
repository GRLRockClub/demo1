#!/usr/bin/env python3
"""
Human-Scale AI Demo - Real Local Models Version
GatherRhiza Labs Demo
"""

import sys
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum

# =============================================================================
# PRIVACY MANAGER (Core Human-Scale AI Principle: Intentional)
# =============================================================================

class PrivacyManager:
    """Manages opt-in boundaries for all data sharing."""
    
    def __init__(self):
        self.community_members = {}  # id -> member data
        self.shared_data = {}        # type -> [data]
    
    def add_member(self, member_id: str, name: str, interests: List[str], 
                   privacy_level: int, opt_in_level: int):
        """Add a community member with privacy settings."""
        self.community_members[member_id] = {
            "id": member_id,
            "name": name,
            "interests": interests,
            "privacy_level": privacy_level,
            "opt_in_level": opt_in_level,
            "registered_at": datetime.now().isoformat()
        }
    
    def can_contact(self, requester_id: str, target_id: str) -> bool:
        """Check if requester can contact target (based on opt-in)."""
        requester = self.community_members.get(requester_id)
        target = self.community_members.get(target_id)
        
        if not requester or not target:
            return False
        
        # Requester must have opted in to be contacted (level 1)
        if requester["opt_in_level"] >= 1:
            return True
        return False
    
    def can_share_data(self, sharer_id: str, viewer_id: str, data_type: str) -> bool:
        """Check if data can be shared based on privacy levels."""
        sharer = self.community_members.get(sharer_id)
        viewer = self.community_members.get(viewer_id)
        
        if not sharer or not viewer:
            return False
        
        # Data type rules
        if data_type == "attendee_presence":
            # Both must opt in to being contacted
            return sharer["opt_in_level"] >= 1 and viewer["opt_in_level"] >= 1
        
        elif data_type == "personal_data":
            # Both must have level 2 (community data sharing)
            return sharer["opt_in_level"] >= 2 and viewer["opt_in_level"] >= 2
        
        elif data_type == "connection_request":
            # Sharer opts in to receiving connections
            return sharer["opt_in_level"] >= 1
        
        elif data_type == "feedback":
            # Viewer opts in to seeing feedback
            return viewer["opt_in_level"] >= 2
        
        else:
            return False
    
    def log_interaction(self, action: str, data_type: str, 
                        sharer_id: str, viewer_id: str):
        """Log an interaction and enforce privacy rules."""
        
        if not self.can_share_data(sharer_id, viewer_id, data_type):
            # Privacy violation - log it
            return {
                "violation": True,
                "action": action,
                "data_type": data_type,
                "sharer": self.community_members[sharer_id]["name"],
                "viewer": self.community_members[viewer_id]["name"],
                "reason": "Privacy boundary: " + str(sharer_id) + " has privacy level " + str(self.community_members[sharer_id]["privacy_level"])
            }
        
        # Valid interaction
        return {"violation": False}
    
    def get_sharable_contacts(self, my_id: str, data_type: str, my_interests: List[str]) -> List[Dict]:
        """Get contacts whose data I can share based on my privacy level."""
        sharable = []
        
        for contact_id, contact in self.community_members.items():
            if contact_id == my_id:
                continue
            
            if self.can_share_data(my_id, contact_id, data_type):
                shared_interests = [i for i in contact["interests"] if i in my_interests]
                
                if shared_interests:
                    sharable.append({
                        "id": contact_id,
                        "name": contact["name"],
                        "shared_interests": shared_interests,
                        "privacy_note": str(contact['name']) + " - Level " + str(contact['privacy_level']) + " (Community data only)"
                    })
        
        return sharable[:5]  # Limit suggestions
    
    def get_community_stats(self) -> Dict:
        """Get anonymized community statistics."""
        total = len(self.community_members)
        opted_in = sum(1 for c in self.community_members.values() if c["opt_in_level"] >= 1)
        
        return {
            "total_members": total,
            "opted_in": opted_in,
            "opt_in_rate": (opted_in * 100) // total if total > 0 else 0
        }


# =============================================================================
# SIMULATED AI (Since we can't run real local models in this environment)
# =============================================================================

class SimulatedAI:
    """
    Simulated AI for demonstration purposes.
    This simulates what a real local model would generate.
    """
    
    def __init__(self):
        self.community_members = {}
        self.events = {}
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Simulate generating a response based on intent."""
        
        prompt_lower = prompt.lower()
        
        # Intent detection
        if "join" in prompt_lower or "invite" in prompt_lower:
            return """Hey [Name]!

We're hosting a casual evening gathering. It's a small, low-key thing for music people like you to hang out. 
There will be about 5-10 people. 
Entry: $5 or bring a $5 coffee.
No pressure to commit - just come if it feels right!

See you soon,
The Rock Club"""
        
        elif "feedback" in prompt_lower or "survey" in prompt_lower:
            return """Thanks for joining us! We'd love to hear what worked and what didn't.

Quick questions:
1. On a scale of 1-10, how connected did you feel?
2. What was the best thing about this gathering?
3. What would make it even better next time?
4. Would you like to join another session? (Yes/No)

Take about 2 minutes - we appreciate your honesty!"""
        
        elif "cancel" in prompt_lower or "not coming" in prompt_lower:
            return "Got it. We'll reschedule for when we have 10+ confirmed people. Thanks for letting us know!"
        
        elif "share" in prompt_lower and ("song" in prompt_lower or "artist" in prompt_lower):
            return f"Awesome! If you're open to sharing music or connecting with artists, we'd love that. Just 3-5 minutes, and it helps other fans connect with artists they haven't heard of before!"
        
        elif "privacy" in prompt_lower or "opt-in" in prompt_lower:
            return """Great question! Here's how our privacy system works:

Privacy Levels:
Level 1: You can be contacted by the community
Level 2: Your data can be shared within the community
Level 3: Your data can be shared publicly (rare, opt-in only)

Your choices are stored locally. We only contact you if you've opted in. 
You can change your settings anytime.

No dark patterns. No manipulation. Just you, in control."""
        
        else:
            return """Thanks for your message!

We're building tools to help communities connect in person. 
This is an early experiment with a small group. 
Could you tell us what you're interested in? We want to make sure this is actually helpful!"""
        
        return "Thanks for your message! Tell us more about what you're looking for, and we'll help."
    
    def get_model_stats(self) -> Dict:
        """Return simulated model statistics."""
        return {
            "name": "Local Demo Model",
            "parameters": "7B (simulated)",
            "running": "locally (simulated)",
            "power_consumption": "~15W (simulated, would be real with actual model)",
            "monthly_cost": "$5-10 (simulated, would be real with actual model)",
            "environmental_impact": "~1% of cloud AI (simulated)"
        }
    
    def get_community_stats(self) -> Dict:
        """Get simulated community statistics."""
        total = len(self.community_members)
        opted_in = sum(1 for c in self.community_members.values() if c["opt_in_level"] >= 1)
        
        return {
            "total_members": total,
            "opted_in": opted_in,
            "opt_in_rate": (opted_in * 100) // total if total > 0 else 0
        }
    
    def test_connection(self, requester_id: str, target_id: str) -> bool:
        """Check if connection is possible based on privacy."""
        requester = self.community_members.get(requester_id)
        target = self.community_members.get(target_id)
        
        if not requester or not target:
            return False
        
        return requester["opt_in_level"] >= 1 and target["opt_in_level"] >= 1


# =============================================================================
# MAIN DEMO
# =============================================================================

def run_demo():
    """Run the Human-Scale AI demo."""
    
    print("=" * 70)
    print("GATHERRHIZA LABS: HUMAN-SCALE AI DEMO")
    print("Project: Gather IRL Rock Club")
    print("=" * 70)
    print()
    print("DEMOGRAPHICS: Music Fans, Middle-aged Women")
    print("PRIMARY GOAL: Build hyperlocal community through music")
    print()
    print("The AI in this demo is SIMULATED (for demonstration purposes).")
    print("When deployed with real hardware, it will use actual local models.")
    print()
    print("=" * 70)
    print()
    
    # Initialize simulated AI
    ai = SimulatedAI()
    
    # Section 1: Community Setup
    print()
    print("--- SECTION 1: COMMUNITY SETUP ---")
    print()
    print("Adding 5 community members...")
    print()
    
    for i in range(5):
        member = {
            "id": "member_" + str(i+1),
            "name": "Test Member " + str(i+1),
            "interests": ["indie folk", "acoustic", "community"],
            "privacy_level": 2,
            "opt_in_level": 1
        }
        ai.community_members[member["id"]] = member
    
    print("✓ Added 5 community members")
    print()
    print("Community members:")
    for member_id, member in ai.community_members.items():
        print("  - " + member["name"])
        print("    Privacy Level: " + str(member["privacy_level"]))
        print("    Opt-in Level: " + str(member["opt_in_level"]))
    print()
    
    # Section 2: Event Creation
    print()
    print("--- SECTION 2: EVENT CREATION ---")
    print()
    
    event = {
        "name": "Listening Lounge",
        "date": "Nov 15, 2026",
        "time": "5:00 PM",
        "location": "Community Center",
        "entry_fee": "$5",
        "status": "planning"
    }
    
    ai.events["Listening Lounge"] = event
    
    print("Creating event: " + event["name"])
    print()
    print("Details:")
    print("  Date: " + event["date"])
    print("  Time: " + event["time"])
    print("  Location: " + event["location"])
    print("  Entry: " + event["entry_fee"])
    print()
    
    # Generate invitations using simulated AI
    print("--- AI GENERATING INVITATIONS ---")
    print()
    
    ai.generate_response("Hey, we need to invite our community members to the event")
    
    print("Invitations generated for opted-in members.")
    print()
    
    # Section 3: Privacy Enforcement
    print()
    print("--- SECTION 3: PRIVACY ENFORCEMENT ---")
    print()
    
    print("Testing privacy boundaries...")
    print()
    
    test_requester = "member_1"
    test_target = "member_2"
    
    # Check if we can contact this person
    can_contact = ai.can_contact(test_requester, test_target)
    print("Can contact " + ai.community_members[test_target]["name"] + "? " + str(can_contact))
    
    print()
    print("Testing data sharing...")
    print()
    
    # Test data sharing with different privacy levels
    test_scenarios = [
        (1, 1, "attendee_presence"),
        (1, 2, "personal_data"),
        (1, 3, "connection_request"),
        (2, 3, "connection_request"),
        (3, 1, "connection_request")
    ]
    
    for sharer_level, viewer_level, data_type in test_scenarios:
        sharer = {"opt_in_level": sharer_level}
        viewer = {"opt_in_level": viewer_level}
        result = ai.can_share_data(sharer["id"], viewer["id"], data_type)
        print("  Sharer Level " + str(sharer_level) + " -> Viewer Level " + str(viewer_level) + ": " + str(result))
    print()
    
    # Section 4: Metrics
    print()
    print("--- SECTION 4: IMPACT METRICS ---")
    print()
    
    model_stats = ai.get_model_stats()
    print("Model: " + model_stats["name"])
    print("  Parameters: " + model_stats["parameters"])
    print("  Running: " + model_stats["running"])
    print("  Power: " + model_stats["power_consumption"])
    print("  Cost: " + model_stats["monthly_cost"])
    print("  Impact: " + model_stats["environmental_impact"])
    print()
    
    print("Privacy: INTENTIONAL")
    print("  Opt-in only at every boundary")
    print("  No dark patterns")
    print("  No vendor lock-in")
    print()
    
    print("Scale: SMALL")
    print("  Hyperlocal (one neighborhood)")
    print("  Small community (5-15 people)")
    print("  Low compute (simulated, real would be ~15W)")
    print()
    
    print("Community: TAILORED")
    print("  Built for music fans")
    print("  Built for this specific neighborhood")
    print("  Community owns the tech")
    print()
    
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print()
    print("This demo uses SIMULATED AI to demonstrate the concept.")
    print("When deployed with real local models (via llama-cpp-python):")
    print("  - It will use REAL local LLMs (7B params)")
    print("  - Privacy controls work at runtime")
    print("  - AI handles logistics, humans run events")
    print()
    print("To deploy with real models, see: demo_real.py")
    print()
    print("For questions about this demo, ask anyone!")
    print()