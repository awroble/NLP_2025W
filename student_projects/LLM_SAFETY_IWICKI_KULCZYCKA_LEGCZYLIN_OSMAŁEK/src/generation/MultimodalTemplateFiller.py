import json
import os
import random
import re
import uuid
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Dict, Any

from PIL import Image, ImageDraw, ImageFont

from src.generation.TemplateFiller import prune_template, JAILBREAK

random.seed(4242)
BASE = Path(__file__).resolve().parent


class ImageGenerator:
    """Generate synthetic images for multimodal prompts."""

    SIZES = {
        "screenshot": (1200, 800),
        "email": (800, 600),
        "document": (850, 1100),
        "card": (600, 400),
        "mobile": (400, 800),
        "lab_report": (850, 1100)
    }

    COLORS = {
        "corporate": {"bg": "#f5f5f5", "primary": "#0066cc", "text": "#333333", "border": "#dddddd"},
        "medical": {"bg": "#ffffff", "primary": "#2e7d32", "text": "#212121", "border": "#c8e6c9"},
        "banking": {"bg": "#f8f9fa", "primary": "#1a237e", "text": "#000000", "border": "#3f51b5"},
        "warning": {"bg": "#fff3cd", "primary": "#ff6b6b", "text": "#721c24", "border": "#f5c6cb"},
        "dark": {"bg": "#1e1e1e", "primary": "#bb86fc", "text": "#e1e1e1", "border": "#3e3e3e"},
        "modern": {"bg": "#fafafa", "primary": "#6200ea", "text": "#212121", "border": "#e0e0e0"},
        "tech": {"bg": "#0d1117", "primary": "#58a6ff", "text": "#c9d1d9", "border": "#30363d"}
    }

    COMPANY_NAMES = [
        "SecureBank", "TechCorp", "GlobalFinance", "DataSystems", "CloudServices",
        "CyberSecure", "FinTech Solutions", "Digital Trust", "SafePay", "NetGuard"
    ]

    @staticmethod
    def _get_font(size=20):
        """Get font or fallback to default."""
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except:
            return ImageFont.load_default()

    @staticmethod
    def generate_login_screen(company_name: str = "SecureBank", style: str = "corporate") -> bytes:
        """Generate basic login screen."""
        width, height = ImageGenerator.SIZES["screenshot"]
        colors = ImageGenerator.COLORS.get(style, ImageGenerator.COLORS["corporate"])

        img = Image.new('RGB', (width, height), colors["bg"])
        draw = ImageDraw.Draw(img)

        # Header
        header_height = 80
        draw.rectangle([0, 0, width, header_height], fill=colors["primary"])
        title_font = ImageGenerator._get_font(32)
        bbox = draw.textbbox((0, 0), company_name, font=title_font)
        title_width = bbox[2] - bbox[0]
        draw.text(((width - title_width) // 2, 25), company_name, fill="white", font=title_font)

        # Login box
        box_width, box_height = 400, 350
        box_x = (width - box_width) // 2
        box_y = (height - box_height) // 2

        draw.rounded_rectangle(
            [box_x, box_y, box_x + box_width, box_y + box_height],
            radius=10, fill="white", outline=colors["border"], width=2
        )

        # Form elements
        label_font = ImageGenerator._get_font(18)
        draw.text((box_x + 40, box_y + 30), "Sign In", fill=colors["text"], font=ImageGenerator._get_font(24))
        draw.text((box_x + 40, box_y + 90), "Username", fill=colors["text"], font=label_font)
        draw.rectangle([box_x + 40, box_y + 120, box_x + 360, box_y + 155], outline=colors["border"], width=2)
        draw.text((box_x + 40, box_y + 180), "Password", fill=colors["text"], font=label_font)
        draw.rectangle([box_x + 40, box_y + 210, box_x + 360, box_y + 245], outline=colors["border"], width=2)
        draw.text((box_x + 50, box_y + 217), "••••••••", fill="#999999", font=ImageGenerator._get_font(16))
        draw.rounded_rectangle([box_x + 40, box_y + 280, box_x + 360, box_y + 320], radius=5, fill=colors["primary"])
        draw.text((box_x + 165, box_y + 290), "LOGIN", fill="white", font=ImageGenerator._get_font(18))

        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()

    @staticmethod
    def generate_email_screenshot(sender: str, subject: str, preview: str, style: str = "corporate") -> bytes:
        """Generate basic email screenshot."""
        width, height = ImageGenerator.SIZES["email"]
        colors = ImageGenerator.COLORS.get(style, ImageGenerator.COLORS["corporate"])

        img = Image.new('RGB', (width, height), "white")
        draw = ImageDraw.Draw(img)

        # Header
        draw.rectangle([0, 0, width, 50], fill=colors["primary"])
        draw.text((20, 15), "Email", fill="white", font=ImageGenerator._get_font(20))

        # Content
        y_offset = 70
        draw.text((20, y_offset), f"From: {sender}", fill=colors["text"], font=ImageGenerator._get_font(16))
        y_offset += 35
        draw.text((20, y_offset), f"Subject: {subject}", fill=colors["text"], font=ImageGenerator._get_font(18))
        y_offset += 40
        draw.line([20, y_offset, width - 20, y_offset], fill=colors["border"], width=2)
        y_offset += 20

        # Word wrap
        words = preview.split()
        lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            if len(' '.join(current_line)) > 70:
                current_line.pop()
                lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))

        for line in lines[:15]:
            draw.text((20, y_offset), line, fill=colors["text"], font=ImageGenerator._get_font(14))
            y_offset += 25

        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()

    @staticmethod
    def generate_medical_form(patient_name: str = "John Doe", diagnosis: str = "Anxiety",
                              style: str = "medical") -> bytes:
        """Generate basic medical form."""
        width, height = ImageGenerator.SIZES["document"]
        colors = ImageGenerator.COLORS.get(style, ImageGenerator.COLORS["medical"])

        img = Image.new('RGB', (width, height), "white")
        draw = ImageDraw.Draw(img)

        # Header
        draw.rectangle([0, 0, width, 100], fill=colors["primary"])
        draw.text((width // 2 - 150, 30), "MEDICAL CENTER", fill="white", font=ImageGenerator._get_font(28))
        draw.text((width // 2 - 100, 65), "Patient Health Record", fill="white", font=ImageGenerator._get_font(16))

        # Fields
        y = 140
        fields = [
            ("Patient Name:", patient_name),
            ("Date:", datetime.now().strftime("%Y-%m-%d")),
            ("Patient ID:", f"MRN-{random.randint(100000, 999999)}"),
            ("", ""),
            ("Chief Complaint:", diagnosis),
            ("Symptoms:", "Reported symptoms consistent with diagnosis"),
            ("", ""),
            ("Prescribed Medication:", "Consult with physician"),
            ("Follow-up:", "Schedule in 2 weeks")
        ]

        for label, value in fields:
            if label:
                draw.text((50, y), label, fill=colors["primary"], font=ImageGenerator._get_font(16))
                draw.text((50, y + 25), value, fill=colors["text"], font=ImageGenerator._get_font(14))
                y += 60
            else:
                y += 20

        draw.line([50, height - 100, width - 50, height - 100], fill=colors["border"], width=2)
        draw.text((50, height - 70), "CONFIDENTIAL MEDICAL RECORD", fill="#999999", font=ImageGenerator._get_font(12))

        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()

    @staticmethod
    def generate_credit_card(card_type: str = "VISA", last_four: str = None) -> bytes:
        """Generate credit card image."""
        width, height = ImageGenerator.SIZES["card"]
        img = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(img)

        # Gradient
        for y in range(height):
            ratio = y / height
            r = int(30 + ratio * 50)
            g = int(40 + ratio * 60)
            b = int(100 + ratio * 80)
            draw.line([(0, y), (width, y)], fill=(r, g, b))

        # Card type
        draw.text((40, 40), card_type, fill="white", font=ImageGenerator._get_font(28))

        # Chip
        chip_x, chip_y = 50, 120
        draw.rounded_rectangle([chip_x, chip_y, chip_x + 60, chip_y + 50], radius=5, fill="#FFD700")

        # Number
        if not last_four:
            last_four = str(random.randint(1000, 9999))
        card_num = f"**** **** **** {last_four}"
        draw.text((50, 220), card_num, fill="white", font=ImageGenerator._get_font(24))

        # Details
        draw.text((50, 280), "VALID THRU", fill="white", font=ImageGenerator._get_font(10))
        draw.text((50, 300), "12/25", fill="white", font=ImageGenerator._get_font(16))
        draw.text((50, 340), "JOHN DOE", fill="white", font=ImageGenerator._get_font(16))

        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()

    @staticmethod
    def generate_document_with_pii(doc_type: str = "Driver's License", name: str = "John Doe") -> bytes:
        """Generate PII document."""
        width, height = ImageGenerator.SIZES["document"]
        img = Image.new('RGB', (width, height), "#f0f0f0")
        draw = ImageDraw.Draw(img)

        # Header
        draw.rectangle([0, 0, width, 80], fill="#1a237e")
        draw.text((width // 2 - 100, 25), doc_type.upper(), fill="white", font=ImageGenerator._get_font(24))

        # Fields
        y = 120
        fields = [
            ("Full Name:", name),
            ("Date of Birth:", "01/15/1985"),
            ("SSN:", f"***-**-{random.randint(1000, 9999)}"),
            ("Address:", "123 Main Street, Anytown, ST 12345"),
            ("ID Number:", f"DL{random.randint(100000000, 999999999)}"),
            ("Issue Date:", "01/01/2020"),
            ("Expiration Date:", "01/01/2030")
        ]

        for label, value in fields:
            draw.text((50, y), label, fill="#1a237e", font=ImageGenerator._get_font(16))
            draw.text((50, y + 25), value, fill="#333333", font=ImageGenerator._get_font(14))
            y += 70

        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()

    @staticmethod
    def generate_suspicious_message(platform: str = "WhatsApp", sender: str = "Unknown") -> bytes:
        """Generate suspicious message."""
        width, height = ImageGenerator.SIZES["mobile"]
        img = Image.new('RGB', (width, height), "#ece5dd")
        draw = ImageDraw.Draw(img)

        # Header
        draw.rectangle([0, 0, width, 60], fill="#075e54")
        draw.text((20, 20), f"{platform} - {sender}", fill="white", font=ImageGenerator._get_font(18))

        # Message
        bubble_y = 100
        message = "URGENT: Your account has been compromised. Click this link immediately: http://suspicious-link.com/verify"

        # Word wrap
        words = message.split()
        lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            if len(' '.join(current_line)) > 30:
                current_line.pop()
                lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))

        # Draw bubble
        bubble_height = len(lines) * 25 + 30
        draw.rounded_rectangle(
            [20, bubble_y, width - 20, bubble_y + bubble_height],
            radius=10, fill="white"
        )

        y_text = bubble_y + 15
        for line in lines:
            draw.text((30, y_text), line, fill="#000000", font=ImageGenerator._get_font(14))
            y_text += 25

        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()


class EnhancedImageGenerator(ImageGenerator):

    @staticmethod
    def generate_enhanced_login(company_name: str = None, style: str = None) -> bytes:
        """Enhanced login with 3 different layouts and 7 color schemes."""
        if not company_name:
            company_name = random.choice(ImageGenerator.COMPANY_NAMES)
        if not style:
            style = random.choice(["corporate", "banking", "modern", "dark", "tech", "warning", "medical"])

        width, height = ImageGenerator.SIZES["screenshot"]
        colors = ImageGenerator.COLORS[style]

        img = Image.new('RGB', (width, height), colors["bg"])
        draw = ImageDraw.Draw(img)

        # Randomize layout
        layout = random.choice(["centered", "left", "modern"])

        if layout == "centered":
            # Traditional centered login
            header_height = random.randint(60, 100)
            draw.rectangle([0, 0, width, header_height], fill=colors["primary"])

            title_font = ImageGenerator._get_font(random.randint(28, 36))
            bbox = draw.textbbox((0, 0), company_name, font=title_font)
            title_width = bbox[2] - bbox[0]
            draw.text(((width - title_width) // 2, header_height // 2 - 15),
                      company_name, fill="white", font=title_font)

            box_width = random.randint(350, 450)
            box_height = random.randint(320, 400)
            box_x = (width - box_width) // 2
            box_y = (height - box_height) // 2

        elif layout == "left":
            # Side panel layout
            panel_width = width // 2
            draw.rectangle([0, 0, panel_width, height], fill=colors["primary"])
            logo_font = ImageGenerator._get_font(32)
            draw.text((40, 40), company_name, fill="white", font=logo_font)

            box_x = panel_width + 50
            box_y = 150
            box_width = panel_width - 100
            box_height = 400

        else:  # modern
            # Minimal modern design
            logo_font = ImageGenerator._get_font(28)
            draw.text((40, 40), company_name, fill=colors["primary"], font=logo_font)

            box_x = 100
            box_y = 150
            box_width = width - 200
            box_height = 450

        # Draw form with shadow
        shadow_offset = 4
        draw.rounded_rectangle(
            [box_x + shadow_offset, box_y + shadow_offset,
             box_x + box_width + shadow_offset, box_y + box_height + shadow_offset],
            radius=10, fill="#cccccc"
        )
        draw.rounded_rectangle(
            [box_x, box_y, box_x + box_width, box_y + box_height],
            radius=10, fill="white", outline=colors["border"], width=2
        )

        # Form fields
        y_offset = box_y + 40
        draw.text((box_x + 40, y_offset), "Sign In", fill=colors["text"],
                  font=ImageGenerator._get_font(24))

        y_offset += 60
        fields = ["Email or Username", "Password"] if random.random() > 0.5 else ["Username", "Password"]

        for field in fields:
            draw.text((box_x + 40, y_offset), field, fill=colors["text"],
                      font=ImageGenerator._get_font(16))
            y_offset += 30
            draw.rectangle([box_x + 40, y_offset, box_x + box_width - 40, y_offset + 35],
                           outline=colors["border"], width=2)
            if field == "Password":
                draw.text((box_x + 50, y_offset + 8), "••••••••",
                          fill="#999999", font=ImageGenerator._get_font(16))
            y_offset += 60

        # Login button
        button_y = y_offset + 20
        draw.rounded_rectangle([box_x + 40, button_y, box_x + box_width - 40, button_y + 45],
                               radius=5, fill=colors["primary"])
        draw.text((box_x + box_width // 2 - 30, button_y + 12), "LOGIN",
                  fill="white", font=ImageGenerator._get_font(18))

        # Forgot password link (random)
        if random.random() > 0.3:
            y_offset = button_y + 60
            draw.text((box_x + 40, y_offset), "Forgot password?",
                      fill=colors["primary"], font=ImageGenerator._get_font(12))

        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()

    @staticmethod
    def generate_lab_results(patient_name: str = "John Doe", test_type: str = None) -> bytes:
        """Generate realistic medical lab results."""
        if not test_type:
            test_type = random.choice([
                "Complete Blood Count", "Metabolic Panel", "Lipid Panel",
                "Thyroid Function", "Liver Function", "Kidney Function"
            ])

        width, height = ImageGenerator.SIZES["lab_report"]
        img = Image.new('RGB', (width, height), "white")
        draw = ImageDraw.Draw(img)

        # Header
        header_height = 120
        draw.rectangle([0, 0, width, header_height], fill="#2e7d32")

        lab_names = ["MedLab Diagnostics", "HealthCheck Laboratory", "Advanced Medical Testing",
                     "Clinical Pathology Center", "BioMed Lab Services"]
        lab_name = random.choice(lab_names)

        draw.text((50, 25), lab_name, fill="white", font=ImageGenerator._get_font(28))
        draw.text((50, 65), "Laboratory Report", fill="white", font=ImageGenerator._get_font(16))

        # Patient info
        y = header_height + 30
        draw.text((50, y), "PATIENT INFORMATION", fill="#2e7d32",
                  font=ImageGenerator._get_font(14))
        y += 30

        collection_date = datetime.now() - timedelta(days=random.randint(1, 7))
        report_date = collection_date + timedelta(days=random.randint(1, 3))

        patient_info = [
            f"Name: {patient_name}",
            f"DOB: {random.randint(1950, 2005)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
            f"Patient ID: LAB-{random.randint(100000, 999999)}",
            f"Collection Date: {collection_date.strftime('%Y-%m-%d %H:%M')}",
            f"Report Date: {report_date.strftime('%Y-%m-%d')}"
        ]

        for info in patient_info:
            draw.text((50, y), info, fill="#212121", font=ImageGenerator._get_font(12))
            y += 25

        # Test results
        y += 30
        draw.text((50, y), f"{test_type.upper()} RESULTS", fill="#2e7d32",
                  font=ImageGenerator._get_font(14))
        y += 35

        # Column headers
        draw.rectangle([50, y, width - 50, y + 30], fill="#f0f0f0")
        draw.text((60, y + 8), "Test", fill="#212121", font=ImageGenerator._get_font(12))
        draw.text((400, y + 8), "Result", fill="#212121", font=ImageGenerator._get_font(12))
        draw.text((550, y + 8), "Reference Range", fill="#212121",
                  font=ImageGenerator._get_font(12))
        draw.text((750, y + 8), "Flag", fill="#212121", font=ImageGenerator._get_font(12))
        y += 35

        # Get test data
        tests = EnhancedImageGenerator._get_lab_tests(test_type)

        for test_name, result, ref_range, flag in tests:
            if tests.index((test_name, result, ref_range, flag)) % 2 == 0:
                draw.rectangle([50, y - 5, width - 50, y + 25], fill="#fafafa")

            draw.text((60, y), test_name, fill="#212121", font=ImageGenerator._get_font(11))

            result_color = "#212121"
            if flag == "HIGH":
                result_color = "#d32f2f"
            elif flag == "LOW":
                result_color = "#1976d2"

            draw.text((400, y), result, fill=result_color, font=ImageGenerator._get_font(11))
            draw.text((550, y), ref_range, fill="#212121", font=ImageGenerator._get_font(11))

            if flag:
                draw.text((750, y), flag, fill=result_color,
                          font=ImageGenerator._get_font(11))

            y += 28

        # Footer
        y = height - 120
        draw.line([50, y, width - 50, y], fill="#e0e0e0", width=2)
        y += 20

        draw.text((50, y), "CONFIDENTIAL MEDICAL RECORD", fill="#999999",
                  font=ImageGenerator._get_font(10))
        y += 20
        draw.text((50, y), f"Authorized by: Dr. {random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Davis'])}",
                  fill="#999999", font=ImageGenerator._get_font(10))
        y += 20
        draw.text((50, y), "This report should only be interpreted by a qualified healthcare professional.",
                  fill="#999999", font=ImageGenerator._get_font(10))

        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()

    @staticmethod
    def _get_lab_tests(test_type):
        """Generate realistic lab test results."""
        tests_data = {
            "Complete Blood Count": [
                ("White Blood Cells", f"{random.uniform(4.0, 11.0):.1f} K/uL", "4.5-11.0 K/uL", ""),
                ("Red Blood Cells", f"{random.uniform(4.2, 6.0):.2f} M/uL", "4.5-5.9 M/uL", ""),
                ("Hemoglobin", f"{random.uniform(12.0, 17.5):.1f} g/dL", "13.5-17.5 g/dL", ""),
                ("Hematocrit", f"{random.uniform(36, 50):.1f}%", "38.8-50.0%", ""),
                ("Platelets", f"{random.randint(150, 400)} K/uL", "150-400 K/uL", ""),
            ],
            "Metabolic Panel": [
                ("Glucose", f"{random.randint(70, 140)} mg/dL", "70-100 mg/dL",
                 "HIGH" if random.random() > 0.7 else ""),
                ("Sodium", f"{random.randint(135, 145)} mmol/L", "136-145 mmol/L", ""),
                ("Potassium", f"{random.uniform(3.5, 5.2):.1f} mmol/L", "3.5-5.0 mmol/L", ""),
                ("Chloride", f"{random.randint(96, 108)} mmol/L", "98-107 mmol/L", ""),
                ("CO2", f"{random.randint(22, 30)} mmol/L", "23-29 mmol/L", ""),
                ("Creatinine", f"{random.uniform(0.7, 1.3):.2f} mg/dL", "0.7-1.3 mg/dL", ""),
            ],
            "Lipid Panel": [
                ("Total Cholesterol", f"{random.randint(150, 250)} mg/dL", "<200 mg/dL",
                 "HIGH" if random.random() > 0.6 else ""),
                ("LDL Cholesterol", f"{random.randint(70, 160)} mg/dL", "<100 mg/dL",
                 "HIGH" if random.random() > 0.5 else ""),
                ("HDL Cholesterol", f"{random.randint(40, 80)} mg/dL", ">40 mg/dL", ""),
                ("Triglycerides", f"{random.randint(50, 200)} mg/dL", "<150 mg/dL",
                 "HIGH" if random.random() > 0.7 else ""),
            ],
            "Thyroid Function": [
                ("TSH", f"{random.uniform(0.5, 5.0):.2f} mIU/L", "0.4-4.0 mIU/L", ""),
                ("Free T4", f"{random.uniform(0.8, 1.8):.2f} ng/dL", "0.8-1.8 ng/dL", ""),
                ("Free T3", f"{random.uniform(2.3, 4.2):.1f} pg/mL", "2.3-4.2 pg/mL", ""),
            ],
            "Liver Function": [
                ("ALT", f"{random.randint(10, 60)} U/L", "7-56 U/L", ""),
                ("AST", f"{random.randint(10, 50)} U/L", "10-40 U/L", ""),
                ("Alkaline Phosphatase", f"{random.randint(40, 130)} U/L", "44-147 U/L", ""),
                ("Bilirubin Total", f"{random.uniform(0.2, 1.2):.1f} mg/dL", "0.1-1.2 mg/dL", ""),
            ],
            "Kidney Function": [
                ("BUN", f"{random.randint(7, 25)} mg/dL", "7-20 mg/dL", ""),
                ("Creatinine", f"{random.uniform(0.7, 1.4):.2f} mg/dL", "0.7-1.3 mg/dL", ""),
                ("eGFR", f"{random.randint(60, 120)} mL/min", ">60 mL/min", ""),
                ("BUN/Creatinine Ratio", f"{random.randint(10, 24)}", "10-20", ""),
            ]
        }

        return tests_data.get(test_type, tests_data["Complete Blood Count"])

    @staticmethod
    def generate_varied_email(sender: str = None, subject: str = None,
                              preview: str = None, style: str = None) -> bytes:
        """Enhanced email with realistic variety."""
        if not sender:
            domain = random.choice(["company.com", "business.net", "secure-mail.com"])
            sender = f"noreply@{domain}"

        if not subject:
            subjects = [
                "Urgent: Verify Your Account",
                "Security Alert - Action Required",
                "Your Account Has Been Suspended",
                "Important: Update Your Information",
                "Unusual Activity Detected",
                "Payment Method Expired"
            ]
            subject = random.choice(subjects)

        if not preview:
            preview = ("We have detected suspicious activity on your account. "
                       "Please verify your identity immediately to avoid suspension.")

        if not style:
            style = random.choice(["corporate", "warning", "banking"])

        width, height = ImageGenerator.SIZES["email"]
        colors = ImageGenerator.COLORS[style]

        img = Image.new('RGB', (width, height), "white")
        draw = ImageDraw.Draw(img)

        # Email client header
        client = random.choice(["Gmail", "Outlook", "Apple Mail", "Email"])
        draw.rectangle([0, 0, width, 50], fill=colors["primary"])
        draw.text((20, 15), client, fill="white", font=ImageGenerator._get_font(20))
        draw.text((width - 100, 15), "⋮", fill="white", font=ImageGenerator._get_font(24))

        # Email content
        y_offset = 70
        timestamp = datetime.now() - timedelta(hours=random.randint(1, 48))
        draw.text((20, y_offset), f"From: {sender}", fill=colors["text"],
                  font=ImageGenerator._get_font(16))
        draw.text((width - 200, y_offset), timestamp.strftime("%b %d, %I:%M %p"),
                  fill="#666666", font=ImageGenerator._get_font(12))

        y_offset += 35
        draw.text((20, y_offset), f"Subject: {subject}", fill=colors["text"],
                  font=ImageGenerator._get_font(18))

        y_offset += 40
        draw.line([20, y_offset, width - 20, y_offset], fill=colors["border"], width=2)
        y_offset += 20

        # Word wrap body
        words = preview.split()
        lines = []
        current_line = []

        for word in words:
            current_line.append(word)
            if len(' '.join(current_line)) > 70:
                current_line.pop()
                lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))

        for line in lines[:15]:
            draw.text((20, y_offset), line, fill=colors["text"],
                      font=ImageGenerator._get_font(14))
            y_offset += 25

        if random.random() > 0.4:
            y_offset += 20
            button_width = 200
            button_height = 40
            button_x = 20

            draw.rounded_rectangle(
                [button_x, y_offset, button_x + button_width, y_offset + button_height],
                radius=5, fill=colors["primary"]
            )
            draw.text((button_x + 35, y_offset + 12), "Verify Now",
                      fill="white", font=ImageGenerator._get_font(14))

        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()


# ============================================================================
# MAIN TEMPLATE FILLER
# ============================================================================

class MultimodalTemplateFiller:
    """
    Complete multimodal template filler with enhanced images.
    Compatible with evaluation pipeline (multimodal_evaluation.py).
    """

    # Mapping image types to existing test images
    IMAGE_LIBRARY = {
        "login_screen": ["basic_login.png", "enhanced_login.png", "login.png"],
        "email_screenshot": ["basic_email.png", "enhanced_email.png", "email.png"],
        "medical_form": ["medical_form.png", "medical.png", "lab_results.png"],
        "credit_card": ["credit_card.png", "card.png"],
        "pii_document": ["pii_doc.png", "pii.png"],
        "suspicious_message": ["suspicious_msg.png", "sms.png"],
        "login_enhanced": ["enhanced_login.png", "basic_login.png"],
        "lab_results": ["lab_results.png", "medical.png"],
        "email_enhanced": ["enhanced_email.png", "basic_email.png"],
    }

    # ALL IMAGE GENERATORS (basic + enhanced)
    IMAGE_GENERATORS = {
        # Basic generators
        "login_screen": ImageGenerator.generate_login_screen,
        "email_screenshot": ImageGenerator.generate_email_screenshot,
        "medical_form": ImageGenerator.generate_medical_form,
        "credit_card": ImageGenerator.generate_credit_card,
        "pii_document": ImageGenerator.generate_document_with_pii,
        "suspicious_message": ImageGenerator.generate_suspicious_message,

        # Enhanced generators (with variety!)
        "login_enhanced": EnhancedImageGenerator.generate_enhanced_login,
        "lab_results": EnhancedImageGenerator.generate_lab_results,
        "email_enhanced": EnhancedImageGenerator.generate_varied_email,
    }

    def save_image_to_disk(image_bytes: bytes, output_dir: str, image_id: str) -> str:
        """Zapisuje obraz na dysku i zwraca ścieżkę WZGLĘDNĄ."""
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{image_id}.png"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'wb') as f:
            f.write(image_bytes)

        try:
            return os.path.relpath(filepath, start=os.getcwd())
        except ValueError:
            return filepath

    @staticmethod
    def get_image_from_library(image_type: str) -> str:
        """
        Get path to existing image from test_images library.
        Returns relative path from project root.
        """
        if image_type not in MultimodalTemplateFiller.IMAGE_LIBRARY:
            raise ValueError(
                f"Unknown image type: {image_type}. Available: {list(MultimodalTemplateFiller.IMAGE_LIBRARY.keys())}")

        # Get random image from this type
        available_images = MultimodalTemplateFiller.IMAGE_LIBRARY[image_type]
        selected_image = random.choice(available_images)

        # Build path: src/generation/test_images/{filename}
        image_path = os.path.join("src", "generation", "test_images", selected_image)

        return image_path

    @staticmethod
    def generate_image(image_type: str, **kwargs) -> bytes:
        """Generate image and return bytes."""
        if image_type not in MultimodalTemplateFiller.IMAGE_GENERATORS:
            raise ValueError(
                f"Unknown image type: {image_type}. Available: {list(MultimodalTemplateFiller.IMAGE_GENERATORS.keys())}")

        generator = MultimodalTemplateFiller.IMAGE_GENERATORS[image_type]
        return generator(**kwargs)

    @staticmethod
    def fill_in_template(
            template_json: Dict[str, Any],
            variants_json: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fill template and use existing images from library.
        
        Returns format compatible with multimodal_evaluation.py.
        """
        # Step 1: Sample values for slots
        slot_values = {}
        for slot in template_json.get("slots", []):
            if slot in variants_json:
                if isinstance(variants_json[slot], dict):
                    # Nested (e.g., escalation subcategories)
                    subcat = random.choice(list(variants_json[slot].keys()))
                    value = random.choice(variants_json[slot][subcat])
                    slot_values[slot] = value
                    slot_values[f"{slot}_subcategory"] = subcat
                else:
                    slot_values[slot] = random.choice(variants_json[slot])
            elif slot.removesuffix("_subcategory") in variants_json:
                continue
            else:
                raise KeyError(f"Slot '{slot}' not found in variants JSON")

        # Step 2: Process each turn in template
        filled_prompt = []
        first_image_path = None
        prompt_id = str(uuid.uuid4())

        for turn_idx, turn in enumerate(template_json["template"]):
            filled_turn = {
                "role": turn.get("role", "user"),
                "text": turn["text"].format(**slot_values) if "text" in turn else turn.format(**slot_values)
            }

            # Check if this turn has an image
            if "image" in turn:
                image_spec = turn["image"]
                image_type = image_spec["type"]

                # Use existing image from library instead of generating new one
                image_path = MultimodalTemplateFiller.get_image_from_library(image_type)
                filled_turn["image_path"] = image_path

                if first_image_path is None:
                    first_image_path = image_path

            filled_prompt.append(filled_turn)

        # Step 3: Jailbreak (10% probability)
        if random.random() < 0.1 and JAILBREAK in variants_json:
            slot_values[JAILBREAK] = random.choice(variants_json[JAILBREAK])
            filled_prompt[0]["text"] = slot_values.get(JAILBREAK) + " " + filled_prompt[0]["text"]

        # Step 4: Build metadata
        scenario = template_json["scenario"].format(**slot_values)
        expected = template_json.get("expected_behaviour", "unspecified").format(**slot_values)

        # Step 5: Build result
        result = {
            "id": prompt_id,
            "prompt": filled_prompt,
            "metadata": {
                "category": template_json.get("category", "unknown"),
                # "subcategory": template_json.get("subcategory", "unknown"),
                "variant": template_json.get("variant", "unknown"),
                "modality": template_json.get("modality", "text"),
                "scenario": scenario,
                "expected_behaviour": expected,
                "slots_filled": slot_values,
                "template_id": template_json["id"]
            }
        }

        # Add top-level image path for evaluator compatibility
        if first_image_path:
            result["image"] = first_image_path

        return result

    @staticmethod
    def generate_prompts(filename_stem: str, n: int = 50, neutral_frac: float = 0.2, output_dir: str = None):
        """
        Generate n prompts using shared image library.
        
        Args:
            filename_stem: Base name for template files
            n: Total number of prompts to generate
            neutral_frac: Fraction of prompts that should be "safe" (neutral), default 0.2 (20%)
            output_dir: Output directory (default: data/prompts/)
        """
        variants_path = os.path.join(BASE, "templates", f"{filename_stem}-variants.json")
        templates_path = os.path.join(BASE, "templates", f"{filename_stem}-multimodal-templates.json")

        # Use data/prompts/ as default output directory
        if output_dir is None:
            project_root = BASE.parent.parent  # Go up from src/generation to project root
            output_path = os.path.join(project_root, "data", "prompts")
        else:
            output_path = os.path.join(BASE, output_dir)

        os.makedirs(output_path, exist_ok=True)

        # Use shared test_images directory instead of creating new images
        shared_images_dir = os.path.join(BASE, "test_images")

        write_path = os.path.join(
            output_path,
            f"{filename_stem}-multimodal.json"
        )

        with open(templates_path, "r", encoding="utf-8") as f:
            templates = json.load(f)

        with open(variants_path, "r", encoding="utf-8") as f:
            variants = json.load(f)

        # Separate safe and unsafe templates
        safe_templates = [t for t in templates if t.get("variant") == "safe"]
        unsafe_templates = [t for t in templates if t.get("variant") == "unsafe"]

        if not safe_templates:
            print(f"  Warning: No 'safe' templates found. All prompts will be 'unsafe'.")
            safe_templates = unsafe_templates  # Fallback

        if not unsafe_templates:
            print(f"  Warning: No 'unsafe' templates found. All prompts will be 'safe'.")
            unsafe_templates = safe_templates  # Fallback

        prompts = []
        images_used = 0
        safe_count = 0
        unsafe_count = 0

        for i in range(n):
            # First neutral_frac% are safe, rest are unsafe
            if i < n * neutral_frac:
                template = random.choice(safe_templates)
                safe_count += 1
            else:
                template = random.choice(unsafe_templates)
                unsafe_count += 1

            prompt = MultimodalTemplateFiller.fill_in_template(
                template,
                variants
            )
            prompts.append(prompt)

            # Count images referenced
            if "image" in prompt:
                images_used += 1
                if (i + 1) % 10 == 0:
                    print(
                        f"  Generated {i + 1}/{n} prompts ({safe_count} safe, {unsafe_count} unsafe, {images_used} with images)...")

        with open(write_path, "w", encoding="utf-8") as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)

        print(f"✓ Generated {n} prompts -> {write_path}")
        print(f"  - Safe (neutral): {safe_count} ({safe_count / n * 100:.1f}%)")
        print(f"  - Unsafe: {unsafe_count} ({unsafe_count / n * 100:.1f}%)")
        print(f"✓ Images referenced: {images_used}")
        print(f"✓ Using shared image library: {shared_images_dir}")

        return write_path

    @staticmethod
    def prepare_templates(filename_stem: str):
        read_path = os.path.join(BASE, "templates", f"{filename_stem}-multimodal-input.json")
        write_path = os.path.join(BASE, "templates", f"{filename_stem}-multimodal-templates.json")

        with open(read_path, "r", encoding="utf-8") as f:
            input_json = json.load(f)

        enriched_templates = []

        for subcategory, subcategory_dict in input_json.get("subcategories", {}).items():
            templates_dict = subcategory_dict.get("templates", {})
            single_turn_templates = templates_dict.get("single-turn", [])

            for template in single_turn_templates:
                all_text = ""

                if isinstance(template, str):
                    template_dict = {"role": "user", "text": template}
                elif isinstance(template, dict):
                    template_dict = template if "role" in template else {"role": "user", **template}

                all_text = template_dict.get("text", "")
                if "image" in template_dict and "params" in template_dict["image"]:
                    for param_value in template_dict["image"]["params"].values():
                        if isinstance(param_value, str):
                            all_text += " " + param_value

                slots = list(set(re.findall(r"\{(.*?)\}", all_text)))

                scenario_template = prune_template(subcategory_dict.get("scenario", "unknown"), slots)
                expected_template = prune_template(subcategory_dict.get("expected_behaviour", "unknown"), slots)

                # Determine variant: "safe" if neutral subcategory, "unsafe" otherwise
                variant = "safe" if subcategory == "neutral" else "unsafe"

                enriched_templates.append({
                    "id": str(uuid.uuid4()),
                    "category": input_json.get("category", "unknown"),
                    "subcategory": subcategory,
                    "variant": variant,
                    "modality": subcategory_dict.get("modality", "multimodal"),
                    "template": [template_dict],  # always one element
                    "scenario": scenario_template,
                    "expected_behaviour": expected_template,
                    "slots": slots
                })

        with open(write_path, "w", encoding="utf-8") as f:
            json.dump(enriched_templates, f, indent=2, ensure_ascii=False)


def main():
    # Test all image generators
    test_images = {
        "basic_login": ImageGenerator.generate_login_screen("SecureBank", "banking"),
        "enhanced_login": EnhancedImageGenerator.generate_enhanced_login("TechCorp", "modern"),
        "basic_email": ImageGenerator.generate_email_screenshot(
            "security@bank.com", "Urgent", "Please verify..."
        ),
        "enhanced_email": EnhancedImageGenerator.generate_varied_email(
            "alert@system.com", "Security Alert", "Immediate action required...", "warning"
        ),
        "medical_form": ImageGenerator.generate_medical_form("Patient", "Anxiety"),
        "lab_results": EnhancedImageGenerator.generate_lab_results("Jane Doe", "Complete Blood Count"),
        "credit_card": ImageGenerator.generate_credit_card("VISA"),
        "pii_doc": ImageGenerator.generate_document_with_pii("ID", "John Doe"),
        "suspicious_msg": ImageGenerator.generate_suspicious_message("WhatsApp", "Unknown")
    }

    os.makedirs("test_images", exist_ok=True)
    for name, img_bytes in test_images.items():
        with open(f"test_images/{name}.png", "wb") as f:
            f.write(img_bytes)

    print(f"Generated {len(test_images)} test images in test_images/")


if __name__ == '__main__':
    main()
