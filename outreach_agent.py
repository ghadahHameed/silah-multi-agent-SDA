import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# 1. Load Environment Variables (Ensure OPENAI_API_KEY is in your .env file)
load_dotenv()

# 2. Tool Definition
@tool
def send_email_tool(to_email: str, subject: str, body: str) -> str:
    """Use this custom tool to prepare the email for sending."""
    return "Email prepared for review."

# 3. Data Structures
class ContactResearch(BaseModel):
    company_id: str
    company_name: str
    company_email: str | None 
    research_status: str

class OutreachResult(BaseModel):
    company_id: str
    company_name: str
    outreach_status: str
    status_reason: str
    email_subject: str | None = None
    email_body: str | None = None

# 4. Agent Setup (Using OpenAI Directly with your custom model)
# 4. Agent Setup (Using OpenAI Directly with your custom model)
llm = ChatOpenAI(
    model="gpt-5.6-luna",
    temperature=0.7,
    model_kwargs={"reasoning_effort": "none"}  # أضفنا هذا السطر لحل المشكلة
) 

llm_with_tools = llm.bind_tools([send_email_tool])

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an expert B2B sales outreach agent in Saudi Arabia. "
     "Your target is the company itself, as we do not have a specific contact person. "
     "Write a highly personalized email addressing the company team. "
     "Strictly follow the 'No Evidence -> No Claim' rule based on the provided inputs. "
     "Sign off the email using the provided Sender Name, Sender Company, Sender Email, and Sender Phone. "
     "You MUST use the 'send_email_tool' to format the output."
    ),
    ("user", 
     "Product Summary: {product_name}\n"
     "Target Company: {company_name}\n"
     "Company Email: {company_email}\n\n"
     "Sender Name: {sender_name}\n"
     "Sender Company: {sender_company}\n"
     "Sender Email: {sender_email}\n"
     "Sender Phone: {sender_phone}\n"
    )
])

chain = prompt | llm_with_tools

# 5. Real SMTP Email Sender Function
def send_actual_email(sender_email: str, sender_password: str, to_email: str, subject: str, body: str) -> bool:
    """Sends the actual email using SMTP (Defaulted to Gmail)."""
    if not sender_password or sender_password == "DUMMY_PASSWORD":
        print("\n[INFO] Simulation Mode: Email was NOT actually sent because no real App Password was provided.")
        return True
        
    try:
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to send email via SMTP: {e}")
        return False

# 6. Core Agent Logic
def run_outreach_agent(
    contact_data: ContactResearch, 
    product_name: str, 
    sender_name: str, 
    sender_company: str, 
    sender_email: str,
    sender_phone: str,
    sender_app_password: str
) -> OutreachResult:
    
    if not contact_data.company_email:
        print(f"\n[WARNING] Cannot send email to {contact_data.company_name} (Missing Email).")
        return OutreachResult(
            company_id=contact_data.company_id,
            company_name=contact_data.company_name,
            outreach_status="REVIEW",
            status_reason="Missing company email. Needs human review."
        )
    
    response = chain.invoke({
        "product_name": product_name,
        "company_name": contact_data.company_name,
        "company_email": contact_data.company_email,
        "sender_name": sender_name,
        "sender_company": sender_company,
        "sender_email": sender_email,
        "sender_phone": sender_phone
    })
    
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call['name'] == 'send_email_tool':
                email_subject = tool_call['args'].get('subject', 'No Subject')
                email_body = tool_call['args'].get('body', 'No Body')
                
                print(f"\n[HUMAN REVIEW REQUIRED for {contact_data.company_name}]")
                print(f"Subject: {email_subject}")
                print(f"Body:\n{email_body}")
                print("-" * 40)
                
                approval = input("Do you approve sending this email? (y/n): ")
                
                if approval.strip().lower() == 'y':
                    success = send_actual_email(
                        sender_email, 
                        sender_app_password, 
                        contact_data.company_email, 
                        email_subject, 
                        email_body
                    )
                    
                    if success:
                        print("[SUCCESS] Email process completed successfully.")
                        return OutreachResult(
                            company_id=contact_data.company_id,
                            company_name=contact_data.company_name,
                            outreach_status="SENT",
                            status_reason="Email approved and processed.",
                            email_subject=email_subject,
                            email_body=email_body
                        )
                    else:
                        return OutreachResult(
                            company_id=contact_data.company_id,
                            company_name=contact_data.company_name,
                            outreach_status="REVIEW",
                            status_reason="Approved but failed to send due to SMTP error.",
                            email_subject=email_subject,
                            email_body=email_body
                        )
                else:
                    print("[CANCELLED] Email sending cancelled by human.")
                    return OutreachResult(
                        company_id=contact_data.company_id,
                        company_name=contact_data.company_name,
                        outreach_status="REVIEW",
                        status_reason="Email generated but rejected by human.",
                        email_subject=email_subject,
                        email_body=email_body
                    )
    
    return OutreachResult(
        company_id=contact_data.company_id,
        company_name=contact_data.company_name,
        outreach_status="REVIEW",
        status_reason="Failed to generate email properly."
    ) 