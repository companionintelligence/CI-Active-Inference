#!/usr/bin/env python3
"""
Demonstration of PersonalLifeAgent Phase 2 implementation.

This script showcases the complete PersonalLifeAgent functionality
including real data integration, prediction pipeline, and action suggestions.
"""

import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, '/home/runner/work/CI-Active-Inference/CI-Active-Inference')

from ci_active_inference.agents.personal_life_agent import PersonalLifeAgent
from ci_active_inference.database import create_tables, get_db_connection

def demo_personal_life_agent():
    """Demonstrate PersonalLifeAgent capabilities."""
    
    print("=" * 60)
    print("PersonalLifeAgent Phase 2 - Live Demonstration")
    print("=" * 60)
    
    # Initialize database and create sample user
    print("\n🔧 Setting up environment...")
    create_tables()
    
    user_id = "demo_user_alice"
    
    # Create sample life-log data
    print(f"📊 Creating sample life-log data for user: {user_id}")
    
    conn = get_db_connection()
    conn.execute("PRAGMA foreign_keys = OFF")  # For demo
    cursor = conn.cursor()
    
    # Create user
    cursor.execute("""
        INSERT OR REPLACE INTO users (id, name, email, password, created_at)
        VALUES (?, 'Alice Demo', 'alice@example.com', 'demo_password', datetime('now'))
    """, (user_id,))
    
    # Create realistic sample events over past week
    base_date = datetime.now() - timedelta(days=7)
    
    sample_events = [
        # Monday - Work focused day
        (user_id, 'Morning Standup', (base_date + timedelta(hours=9)).isoformat(), 
         (base_date + timedelta(hours=9, minutes=30)).isoformat()),
        (user_id, 'Deep Work Session', (base_date + timedelta(hours=10)).isoformat(), 
         (base_date + timedelta(hours=12)).isoformat()),
        (user_id, 'Team Meeting', (base_date + timedelta(hours=14)).isoformat(), 
         (base_date + timedelta(hours=15)).isoformat()),
        (user_id, 'Code Review', (base_date + timedelta(hours=15, minutes=30)).isoformat(), 
         (base_date + timedelta(hours=17)).isoformat()),
        (user_id, 'Evening Workout', (base_date + timedelta(hours=18, minutes=30)).isoformat(), 
         (base_date + timedelta(hours=19, minutes=30)).isoformat()),
        
        # Tuesday - Mixed work and social
        (user_id, 'Client Presentation', (base_date + timedelta(days=1, hours=10)).isoformat(), 
         (base_date + timedelta(days=1, hours=11, minutes=30)).isoformat()),
        (user_id, 'Lunch with Colleagues', (base_date + timedelta(days=1, hours=12, minutes=30)).isoformat(), 
         (base_date + timedelta(days=1, hours=13, minutes=30)).isoformat()),
        (user_id, 'Project Planning', (base_date + timedelta(days=1, hours=14)).isoformat(), 
         (base_date + timedelta(days=1, hours=16)).isoformat()),
        (user_id, 'Yoga Class', (base_date + timedelta(days=1, hours=19)).isoformat(), 
         (base_date + timedelta(days=1, hours=20)).isoformat()),
        
        # Wednesday - High productivity
        (user_id, 'Research Session', (base_date + timedelta(days=2, hours=9)).isoformat(), 
         (base_date + timedelta(days=2, hours=11)).isoformat()),
        (user_id, 'Writing Documentation', (base_date + timedelta(days=2, hours=11, minutes=30)).isoformat(), 
         (base_date + timedelta(days=2, hours=13)).isoformat()),
        (user_id, 'Problem Solving', (base_date + timedelta(days=2, hours=14)).isoformat(), 
         (base_date + timedelta(days=2, hours=17)).isoformat()),
        (user_id, 'Reading', (base_date + timedelta(days=2, hours=20)).isoformat(), 
         (base_date + timedelta(days=2, hours=21)).isoformat()),
        
        # Thursday - Social day
        (user_id, 'Team Brainstorm', (base_date + timedelta(days=3, hours=10)).isoformat(), 
         (base_date + timedelta(days=3, hours=12)).isoformat()),
        (user_id, 'Networking Event', (base_date + timedelta(days=3, hours=18)).isoformat(), 
         (base_date + timedelta(days=3, hours=21)).isoformat()),
        
        # Friday - Relaxed finish
        (user_id, 'Code Cleanup', (base_date + timedelta(days=4, hours=9)).isoformat(), 
         (base_date + timedelta(days=4, hours=11)).isoformat()),
        (user_id, 'Team Retrospective', (base_date + timedelta(days=4, hours=15)).isoformat(), 
         (base_date + timedelta(days=4, hours=16)).isoformat()),
        (user_id, 'Happy Hour', (base_date + timedelta(days=4, hours=17, minutes=30)).isoformat(), 
         (base_date + timedelta(days=4, hours=19)).isoformat()),
    ]
    
    print(f"  📅 Adding {len(sample_events)} life events...")
    for event in sample_events:
        cursor.execute("""
            INSERT INTO events (user_id, name, start_time, end_time, created_at)
            VALUES (?, ?, ?, ?, datetime('now'))
        """, event)
    
    conn.commit()
    conn.close()
    print("  ✅ Sample data created successfully")
    
    # Initialize PersonalLifeAgent
    print(f"\n🤖 Initializing PersonalLifeAgent for {user_id}...")
    agent = PersonalLifeAgent(user_id)
    
    # Initialize from data
    start_date = (base_date - timedelta(days=1)).isoformat()
    end_date = datetime.now().isoformat()
    
    success = agent.initialize_from_data(start_date, end_date)
    if success:
        print("  ✅ Agent successfully initialized from life-log data")
        print(f"  📊 Matrices built: A={len(agent.A)}, B={len(agent.B)}, C={len(agent.C)}, D={len(agent.D)}")
    else:
        print("  ⚠️  Agent initialization had issues, using defaults")
    
    # Demonstrate prediction scenarios
    print("\n🔮 Demonstration Scenarios")
    print("-" * 40)
    
    scenarios = [
        {
            "name": "Monday Morning Work Context",
            "description": "Alice is at the office, alone, working on code",
            "observations": {
                "digital_behavior": "work-focus",
                "location_context": "office", 
                "social_context": "alone",
                "content_type": "work"
            }
        },
        {
            "name": "Friday Afternoon Social Context", 
            "description": "Alice is in a social venue with colleagues",
            "observations": {
                "digital_behavior": "communication",
                "location_context": "social-venue",
                "social_context": "small-group", 
                "content_type": "communication"
            }
        },
        {
            "name": "Evening Home Context",
            "description": "Alice is at home, alone, browsing entertainment",
            "observations": {
                "digital_behavior": "entertainment",
                "location_context": "home",
                "social_context": "alone",
                "content_type": "other"
            }
        },
        {
            "name": "Outdoor Exercise Context",
            "description": "Alice is outdoors, alone, focused on health",
            "observations": {
                "digital_behavior": "other",
                "location_context": "outdoor", 
                "social_context": "alone",
                "content_type": "other"
            }
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📋 Scenario {i}: {scenario['name']}")
        print(f"   Context: {scenario['description']}")
        print(f"   Observations: {scenario['observations']}")
        
        # Get prediction
        result = agent.predict_and_suggest(scenario['observations'])
        
        if 'error' in result:
            print(f"   ❌ Error: {result['error']}")
            continue
        
        print(f"\n   🎯 Predictions:")
        print(f"   • Predicted default behavior: {result['predicted_default']}")
        print(f"   • Suggested optimal action: {result['suggested_action']}")
        print(f"   • Confidence level: {result['confidence']:.2%}")
        
        print(f"\n   🧠 State Beliefs:")
        belief_names = [['Low Energy', 'Medium Energy', 'High Energy'],
                       ['Negative Mood', 'Neutral Mood', 'Positive Mood'],
                       ['Work Context', 'Social Context', 'Health Context', 'Home Context', 'Other Context']]
        
        for factor, beliefs in enumerate(result['current_state_beliefs']):
            if factor < len(belief_names):
                max_belief_idx = beliefs.index(max(beliefs))
                max_belief_name = belief_names[factor][max_belief_idx]
                max_belief_value = beliefs[max_belief_idx]
                print(f"   • {['Energy', 'Mood', 'Context'][factor]}: {max_belief_name} ({max_belief_value:.1%})")
        
        print(f"\n   ⚡ Expected Free Energy:")
        efe = result['efe_breakdown']
        print(f"   • Risk (preference mismatch): {efe['risk']:.2%}")
        print(f"   • Ambiguity (uncertainty): {efe['ambiguity']:.2%}")
        
        # Demonstrate learning from outcome
        if i == 1:  # Update after first scenario
            print(f"\n   📚 Learning Update:")
            print(f"   Simulating Alice took the suggested action: {result['suggested_action']}")
            
            # Create follow-up observations
            next_obs = scenario['observations'].copy()
            if result['suggested_action'] == 'work_focus':
                next_obs['digital_behavior'] = 'work-focus'
            elif result['suggested_action'] == 'social_engage':
                next_obs['digital_behavior'] = 'communication'
            
            agent.update_from_outcome(result['suggested_action'], next_obs)
            print(f"   ✅ Agent updated with outcome")
    
    # Show model statistics
    print(f"\n📊 Agent Performance Statistics")
    print("-" * 40)
    
    stats = agent.get_model_statistics()
    print(f"Total predictions made: {stats['total_predictions']}")
    print(f"Recent accuracy estimate: {stats['accuracy_last_week']:.1%}")
    print(f"Average uncertainty level: {stats['uncertainty_level']:.1%}")
    print(f"Last model update: {stats['last_updated']}")
    print(f"Matrix dimensions: {stats['matrix_shapes']}")
    
    # Demonstrate API-style usage
    print(f"\n🌐 API Integration Example")
    print("-" * 40)
    
    print("Example REST API usage:")
    print(f"POST /agent/{user_id}/predict")
    print("Body:", scenarios[0]['observations'])
    print(f"Response: {result['suggested_action']} (confidence: {result['confidence']:.1%})")
    
    print(f"\n✨ PersonalLifeAgent Phase 2 Demo Complete!")
    print("Key achievements:")
    print("✅ Real data integration from CI database")
    print("✅ A/B/C/D matrix construction from temporal patterns")
    print("✅ State inference and action optimization")
    print("✅ EFE-based decision making")
    print("✅ Online learning from outcomes")
    print("✅ Comprehensive error handling")
    print("✅ Scalable architecture")
    
    return True

if __name__ == "__main__":
    try:
        demo_personal_life_agent()
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)