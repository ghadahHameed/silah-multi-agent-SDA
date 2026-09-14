from outreach_agent import ContactResearch, run_outreach_agent

def main():
    my_name = "Aisha"
    my_company = "Silah Solutions"
    my_email = "aisha@silah.com"
    my_phone = "+966500000000"
    my_app_password = "DUMMY_PASSWORD"
    
    product = "Silah: AI Multi-Agent Outbound System"

    stc_data = ContactResearch(
        company_id="SA-001",
        company_name="Saudi Telecom Company (stc)",
        company_email="info@stc.com.sa",
        research_status="Completed"
    )
    
    elm_data = ContactResearch(
        company_id="SA-002",
        company_name="Elm Company",
        company_email=None,
        research_status="Completed"
    )
    
    print("\n--- Test 1: STC ---")
    result1 = run_outreach_agent(
        stc_data, 
        product, 
        my_name, 
        my_company, 
        my_email, 
        my_phone, 
        my_app_password
    )
    print(f"Final Status: {result1.outreach_status}")
    print(f"Saved Email Subject: {result1.email_subject}")
    
    print("\n--- Test 2: Elm ---")
    result2 = run_outreach_agent(
        elm_data, 
        product, 
        my_name, 
        my_company, 
        my_email, 
        my_phone, 
        my_app_password
    )
    print(f"Final Status: {result2.outreach_status}")
    print(f"Saved Email Subject: {result2.email_subject}")

if __name__ == "__main__":
    main() 