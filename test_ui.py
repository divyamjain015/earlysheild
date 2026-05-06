import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import time

# To run this test, ensure the Streamlit app is running on port 8501
# Command to run: pytest test_ui.py -v

@pytest.fixture(scope="module")
def driver():
    """Setup Selenium WebDriver."""
    options = webdriver.ChromeOptions()
    # Uncomment the next line to run headlessly (invisible) for CI/CD pipelines
    # options.add_argument('--headless') 
    
    # Ensure you have the correct webdriver installed for your browser
    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(5)
    
    yield driver
    
    # Teardown: close the browser after tests are done
    driver.quit()

def test_app_loads(driver):
    """Test if the Streamlit app loads correctly."""
    driver.get("http://localhost:8501")
    wait = WebDriverWait(driver, 15)
    
    # Wait for the main title to ensure the app has rendered
    title = wait.until(EC.presence_of_element_located(
        (By.XPATH, "//*[contains(text(), 'Pre-Delinquency Risk Engine')]")
    ))
    assert title.is_displayed()

def test_predict_risk_flow(driver):
    """Test the end-to-end flow of filling out the form and getting a prediction."""
    driver.get("http://localhost:8501")
    wait = WebDriverWait(driver, 15)
    
    # Wait for the app to be fully loaded (the sidebar is a good indicator)
    wait.until(EC.presence_of_element_located((By.XPATH, "//*[@data-testid='stSidebar']")))
    
    # ---------------------------------------------------------
    # Helper function to set text inputs in Streamlit
    # Streamlit inputs are usually inside a div that contains the label
    # ---------------------------------------------------------
    def set_text_input(label_text, value):
        # Streamlit usually places labels inside <p> tags. We also provide an aria-label fallback.
        xpath = f"//div[@data-testid='stTextInput'][.//p[text()='{label_text}']]//input"
        try:
            element = wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
        except:
            xpath_fallback = f"//input[@aria-label='{label_text}']"
            element = wait.until(EC.element_to_be_clickable((By.XPATH, xpath_fallback)))
            
        # Clear existing text using keyboard shortcuts as Streamlit inputs can be stubborn
        element.send_keys(Keys.CONTROL + "a")
        element.send_keys(Keys.DELETE)
        element.send_keys(value)
        element.send_keys(Keys.TAB) # Trigger the update
        time.sleep(0.5) # Give Streamlit time to process the state change

    def set_number_input(label_text, value):
        xpath = f"//div[@data-testid='stNumberInput'][.//p[text()='{label_text}']]//input"
        try:
            element = wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
        except:
            xpath_fallback = f"//input[@aria-label='{label_text}']"
            element = wait.until(EC.element_to_be_clickable((By.XPATH, xpath_fallback)))
            
        element.send_keys(Keys.CONTROL + "a")
        element.send_keys(Keys.DELETE)
        element.send_keys(str(value))
        element.send_keys(Keys.TAB)
        time.sleep(0.5)
        
    # Note: Sliders and Selectboxes in Streamlit are more complex to automate with Selenium 
    # because they don't use standard HTML <select> or <input type="range">. 
    # For this basic test, we will leave the selectboxes/sliders at their defaults, 
    # but we will fill the standard text and number inputs.

    # 1. Fill Input Fields
    set_text_input("Customer ID", "CUST_999")
    set_text_input("Month", "2024-12")
    set_number_input("EMI Amount", 25000)
    set_number_input("Monthly Income", 80000)
    
    # 2. Click the Predict Button
    # Streamlit buttons have the text inside a nested p/div element
    predict_btn_xpath = "//button[descendant::div[contains(text(), 'Predict Risk')]] | //button[descendant::p[contains(text(), 'Predict Risk')]]"
    predict_button = wait.until(EC.element_to_be_clickable((By.XPATH, predict_btn_xpath)))
    predict_button.click()
    
    # 3. Wait for Results to Appear
    # We look for the "Results" subheader
    results_header = wait.until(EC.presence_of_element_located(
        (By.XPATH, "//*[contains(text(), '📊 Results')]")
    ))
    assert results_header.is_displayed()
    
    # 4. Verify specific output elements are present (e.g., 'Risk Bucket' text is generated)
    risk_bucket_metric = wait.until(EC.presence_of_element_located(
        (By.XPATH, "//*[contains(text(), 'Risk Bucket')]")
    ))
    assert risk_bucket_metric.is_displayed()
    
    # Check that SHAP plots are loaded
    shap_header = wait.until(EC.presence_of_element_located(
        (By.XPATH, "//*[contains(text(), 'Model Explainability')]")
    ))
    assert shap_header.is_displayed()
