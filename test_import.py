# test_import.py
from qwen_payslip_processor import QwenPayslipProcessor

print("Package imported successfully")
processor = QwenPayslipProcessor(window_mode="whole", force_cpu=True)
print(f"Device: {processor.device}")