import streamlit as st
import os
import tempfile
import requests
from bs4 import BeautifulSoup
import re
import json
from openai import OpenAI
import pdfplumber
from fpdf import FPDF
import subprocess
import base64
import time

# Setup page configuration
st.set_page_config(page_title="AI Resume Tailor", layout="wide")

# Initialize OpenAI client (using environment variable)
def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    
    if not api_key:
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
        if not api_key:
            st.warning("Please enter an OpenAI API key to continue.")
            st.stop()
            
    return OpenAI(api_key=api_key)

# Improved scrape job description from URL
def scrape_job_description(url, max_retries=3):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0"
    }
    
    # Check if URL is properly formatted
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    for attempt in range(max_retries):
        try:
            # Increase timeout for slow-responding sites
            with st.spinner(f"Fetching job description (attempt {attempt+1}/{max_retries})..."):
                response = requests.get(url, headers=headers, timeout=20)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Try different strategies to find job description
                job_description = ""
                
                # Strategy 1: Look for common job description containers
                job_containers = soup.find_all(['div', 'section'], class_=re.compile(r'(job|position|description|details|posting)', re.I))
                
                # Strategy 2: Look for specific job-related sections
                if not job_description:
                    for section in soup.find_all(['section', 'div'], id=re.compile(r'(job|description|requirements|jd)', re.I)):
                        job_description += section.get_text(separator=' ', strip=True) + "\n"
                
                # Strategy 3: Look for job-related strings in text
                if not job_description:
                    page_text = soup.get_text()
                    job_sections = re.findall(r'(?:Job Description|Requirements|Qualifications|Responsibilities)(?:[\s\S]{10,1000}?)(?=Job Description|Requirements|Qualifications|Responsibilities|\Z)', page_text)
                    job_description = "\n".join(job_sections)
                
                # Strategy 4: Fallback to main content
                if not job_description or len(job_description) < 200:
                    # Try to find main content
                    main_content = soup.find(['main', 'article']) or soup.find('div', class_=re.compile(r'(content|main)', re.I))
                    if main_content:
                        job_description = main_content.get_text(separator=' ', strip=True)
                    else:
                        # Just get body text, removing header and footer
                        body = soup.find('body')
                        if body:
                            # Exclude common navigation and footer elements
                            for nav in body.find_all(['nav', 'header', 'footer']):
                                nav.decompose()
                            job_description = body.get_text(separator=' ', strip=True)
                
                # Clean up the text
                job_description = re.sub(r'\s+', ' ', job_description).strip()
                
                # If we got a reasonable amount of text, consider it successful
                if len(job_description) > 200:
                    return job_description
                
                # If job description is too short, try again
                time.sleep(1)  # Add a small delay between attempts
                
        except requests.exceptions.Timeout:
            st.warning(f"Attempt {attempt+1} timed out. Retrying...")
            time.sleep(2)  # Wait before retrying
        except requests.exceptions.RequestException as e:
            st.warning(f"Error during attempt {attempt+1}: {str(e)}")
            time.sleep(2)  # Wait before retrying
    
    # If all attempts failed, return None
    return None

# Parse uploaded resume
def parse_resume(file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        with pdfplumber.open(tmp_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        os.remove(tmp_path)
        return text.strip()
    except Exception as e:
        st.error(f"Error parsing resume: {str(e)}")
        return None

# GPT resume tailoring
def tailor_resume(client, resume_text, job_description, user_name, user_contact):
    try:
        prompt = f"""
        Rewrite the following resume to match the job description. Make it tailored and optimized for ATS (Applicant Tracking Systems).
        
        JOB DESCRIPTION:
        {job_description}

        CANDIDATE INFO:
        Name: {user_name}
        Contact: {user_contact}

        ORIGINAL RESUME:
        {resume_text}

        INSTRUCTIONS:
        1. Extract relevant skills and experience from the original resume
        2. Reorganize content to highlight experience relevant to the job description
        3. Use keywords from the job description where applicable
        4. Create a summary section that matches the job requirements
        5. Format sections clearly (Experience, Education, Skills, etc.)
        6. Return the resume in plain text format structured to be ATS-friendly
        7. DO NOT invent fake experience or skills
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error during resume tailoring: {str(e)}")
        return None

# Create PDF directly with FPDF (no LaTeX dependency)
def create_pdf(tailored_text, user_name, user_contact):
    try:
        pdf = FPDF(orientation='P', unit='mm', format='A4')
        pdf.add_page()
        
        # Try to use DejaVu fonts if available, fall back to built-in fonts if not
        try:
            pdf.add_font('DejaVu', '', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', uni=True)
            pdf.add_font('DejaVu', 'B', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', uni=True)
            font_family = 'DejaVu'
        except:
            # Fall back to built-in fonts if DejaVu isn't available
            font_family = 'Arial'
        
        # Header with name
        pdf.set_font(font_family, 'B', 16)
        pdf.cell(0, 10, user_name, ln=True, align='C')
        
        # Contact info
        pdf.set_font(font_family, '', 10)
        pdf.cell(0, 5, user_contact, ln=True, align='C')
        pdf.ln(5)
        
        # Divider line
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        # Resume content
        pdf.set_font(font_family, '', 10)
        
        # Process the tailored text by sections
        sections = tailored_text.split('\n\n')
        for section in sections:
            if section.strip():
                # Check if it's a section header (all caps or ending with :)
                if section.isupper() or section.strip().endswith(':'):
                    pdf.set_font(font_family, 'B', 12)
                    pdf.cell(0, 6, section.strip(), ln=True)
                    pdf.set_font(font_family, '', 10)
                else:
                    # Handle multi-line sections
                    lines = section.split('\n')
                    for line in lines:
                        if line.strip():
                            # Handle potential encoding issues
                            try:
                                pdf.multi_cell(0, 5, line.strip())
                            except:
                                # If there's an encoding issue, try to normalize the text
                                cleaned_line = ''.join(c for c in line if ord(c) < 128)
                                pdf.multi_cell(0, 5, cleaned_line.strip())
                pdf.ln(3)
        
        return pdf.output(dest='S').encode('latin1', errors='replace')
    except Exception as e:
        st.error(f"Error creating PDF: {str(e)}")
        # Fallback to text file if PDF creation fails
        return tailored_text.encode('utf-8')

# Main app UI
def main():
    st.title("📄 Touchbase Technologies Resume Tailoring Bot")
    st.write("Upload your resume and enter a job posting URL to create a tailored resume")
    
    with st.expander("How to use", expanded=False):
        st.markdown("""
        1. Enter your name and contact information
        2. Upload your current resume (PDF format)
        3. Enter the URL of the job posting you're applying for
        4. Click 'Tailor My Resume' to generate a customized version
        5. Download your tailored resume as PDF
        """)
    
    # User information
    col1, col2 = st.columns(2)
    with col1:
        user_name = st.text_input("Your Full Name")
    with col2:
        user_contact = st.text_input("Contact Info (email, phone, location)")
    
    # File uploader and URL input
    uploaded_file = st.file_uploader("📄 Upload your Resume (PDF)", type=["pdf"])
    
    url_col, manual_col = st.columns(2)
    with url_col:
        job_url = st.text_input("🔗 Job Posting URL")
    with manual_col:
        manual_option = st.checkbox("Or paste job description manually")
    
    manual_description = ""
    if manual_option:
        manual_description = st.text_area("Job Description:", height=200)
    
    if st.button("✨ Tailor My Resume"):
        if not user_name or not user_contact:
            st.error("Please provide your name and contact information.")
        elif not uploaded_file:
            st.error("Please upload your resume.")
        elif not job_url and not manual_description:
            st.error("Please provide either a job posting URL or paste a job description.")
        else:
            with st.spinner("Analyzing your information..."):
                # Initialize OpenAI client
                client = get_openai_client()
                
                # Get job description (from URL or manual input)
                job_description = ""
                if manual_description:
                    job_description = manual_description
                else:
                    job_description = scrape_job_description(job_url)
                    
                    if not job_description:
                        st.error("Failed to extract job description. Please check the URL or paste the job description manually.")
                        job_description = st.text_area("Paste job description manually:", height=200)
                        if not job_description:
                            st.stop()
                
                # Display job description preview for verification
                with st.expander("Job Description Preview", expanded=False):
                    st.text_area("", job_description, height=200)
                    if st.button("Edit Job Description"):
                        job_description = st.text_area("Edit job description:", value=job_description, height=300)
                
                # Parse resume
                resume_text = parse_resume(uploaded_file)
                if not resume_text:
                    st.error("Failed to parse your resume. Please check the file.")
                    st.stop()
                
                # Get tailored resume
                with st.spinner("AI is tailoring your resume..."):
                    tailored_resume = tailor_resume(client, resume_text, job_description, user_name, user_contact)
                    if not tailored_resume:
                        st.error("Failed to tailor your resume. Please try again.")
                        st.stop()
                
                # Create PDF
                try:
                    with st.spinner("Generating PDF..."):
                        pdf_bytes = create_pdf(tailored_resume, user_name, user_contact)
                    
                    # Show success and offer download
                    st.success("✅ Resume tailored successfully!")
                    
                    # Show text preview
                    with st.expander("Preview Tailored Content", expanded=True):
                        st.text_area("", tailored_resume, height=400)
                    
                    # Download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📥 Download as PDF",
                            data=pdf_bytes,
                            file_name="tailored_resume.pdf",
                            mime="application/pdf"
                        )
                    with col2:
                        st.download_button(
                            label="📄 Download as Text",
                            data=tailored_resume,
                            file_name="tailored_resume.txt",
                            mime="text/plain"
                        )
                    
                except Exception as e:
                    st.error(f"Error in PDF generation: {str(e)}")
                    # Fallback to text-only
                    st.download_button(
                        label="📄 Download as Text",
                        data=tailored_resume,
                        file_name="tailored_resume.txt",
                        mime="text/plain"
                    )

if __name__ == "__main__":
    main()
