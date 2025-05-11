from State import State
from LLM import llm
from Ticket_assignment import create_tables, add_ticket, assign_ticket, add_person

def person_support_needed(state:State):
    
    create_tables()
    
    # add_person("Alice", "alice@support.com", "E_commerce", "login issue, password reset", "2FA, VPN")

    # add_person("Bob", "bob@support.com", "E_commerce", "checkout issues, payment failure", "billing system, payment gateway")

    # add_person("Eve", "eve@support.com", "E_commerce", "cart not updating, session timeout", "server cache, session handling")

    # add_person("Grace", "grace@support.com", "E_commerce", "product search, filters not working", "search engine, UI bugs")

    # add_person("Hannah", "hannah@support.com", "E_commerce", "order not processed", "inventory sync, database issue")

    # add_person("Isabel", "isabel@support.com", "E_commerce", "promotion code invalid", "coupon system, promo validation")

    # add_person("John", "john@support.com", "E_commerce", "account locked", "login attempts, security protocols")

    # add_person("Kylie", "kylie@support.com", "E_commerce", "delivery tracking, order not received", "shipping carrier, tracking system")

    # add_person("Liam", "liam@support.com", "E_commerce", "payment method rejected", "credit card validation, third-party payment processor")

    # add_person("Mia", "mia@support.com", "E_commerce", "discount not applied", "checkout system, pricing rules")

    # # Bank Finance
    # add_person("James", "james@support.com", "Bank_finance", "transaction declined", "bank policy, payment processor")

    # add_person("Kevin", "kevin@support.com", "Bank_finance", "account balance discrepancy", "account syncing, database issue")

    # add_person("Laura", "laura@support.com", "Bank_finance", "missing deposit", "bank transaction records, settlement issue")

    # add_person("Megan", "megan@support.com", "Bank_finance", "card activation issues", "card system, user verification")

    # add_person("Nina", "nina@support.com", "Bank_finance", "fraudulent activity on account", "fraud detection, account monitoring")

    # add_person("Oscar", "oscar@support.com", "Bank_finance", "loan approval status", "loan processing, credit score check")

    # add_person("Paul", "paul@support.com", "Bank_finance", "fund transfer issues", "payment gateway, network error")

    # add_person("Quincy", "quincy@support.com", "Bank_finance", "account locked", "security settings, failed login attempts")

    # add_person("Rebecca", "rebecca@support.com", "Bank_finance", "duplicate transactions", "transaction logs, server issue")

    # add_person("Steve", "steve@support.com", "Bank_finance", "ATM withdrawal issue", "ATM network, local bank branch")

    # # Healthcare
    # add_person("Nancy", "nancy@support.com", "Healthcare", "appointment scheduling error", "appointment system, user interface")

    # add_person("Olivia", "olivia@support.com", "Healthcare", "prescription refill issues", "pharmacy system, patient records")

    # add_person("Peter", "peter@support.com", "Healthcare", "insurance verification problem", "insurance system, verification process")

    # add_person("Quinn", "quinn@support.com", "Healthcare", "patient records access denied", "hospital database, user permissions")

    # add_person("Rachel", "rachel@support.com", "Healthcare", "billing issues, payment denied", "payment processor, billing software")

    # add_person("Sophia", "sophia@support.com", "Healthcare", "emergency contact update", "database sync, user form submission")

    # add_person("Tom", "tom@support.com", "Healthcare", "lab test results missing", "test result database, lab system")

    # add_person("Ursula", "ursula@support.com", "Healthcare", "patient portal login issues", "user authentication, portal security")

    # add_person("Victor", "victor@support.com", "Healthcare", "medication error, prescription mismatch", "pharmacy system, prescription database")

    # add_person("Wendy", "wendy@support.com", "Healthcare", "billing statement clarification", "accounting system, payment history")

    # # Telcom
    # add_person("Charlie", "charlie@support.com", "Tellicom", "firewall, login", "port blocking", availability="yes")

    # add_person("David", "david@support.com", "Tellicom", "slow internet", "network congestion, router issue")

    # add_person("Ella", "ella@support.com", "Tellicom", "call drop, connectivity", "signal tower, mobile network")

    # add_person("Frank", "frank@support.com", "Tellicom", "account suspended", "security check, payment issue")

    # add_person("Grace", "grace@support.com", "Tellicom", "data plan activation", "account settings, server issue")

    # add_person("Hank", "hank@support.com", "Tellicom", "roaming not working", "network coverage, SIM card issue")

    # add_person("Ivy", "ivy@support.com", "Tellicom", "incorrect billing", "billing system, usage tracking")

    # add_person("Jack", "jack@support.com", "Tellicom", "service outage", "network downtime, maintenance")

    # add_person("Kim", "kim@support.com", "Tellicom", "data usage dispute", "usage data, monitoring system")

    # add_person("Leo", "leo@support.com", "Tellicom", "device setup assistance", "SIM card, device compatibility")


    ticket_id = add_ticket("UserX", state["decision"] , state["input"])
    # print("PERSON DETAILS TICKETS ---------------->",assign_ticket(ticket_id))
    return {"support_person_details":assign_ticket(ticket_id)}

def Provide_solution_based_on_feedback(state:State):

    result = llm.invoke(f"Based on the user's initial question: {state['input']}, and the solution provided: {state['solution_providing_output']}, now consider the user's feedback: {state['feedback_by_user']}. Using this information, generate a refined solution that directly addresses the user's feedback, ensuring clarity, completeness, and relevance.")

    return {"solution_providing_output":result.content}