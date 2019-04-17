from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC # available since 2.26.0
from selenium.webdriver.common.by import By
caps = DesiredCapabilities().FIREFOX
import sys

def check_exists_by_xpath(xpath):
   try:
       driver.find_element_by_xpath(xpath)
   except:
       return False
   return True


def scraping_news(URL):
    caps["pageLoadStrategy"] = "normal"
    try:
        delay = 15
        baseURL = URL
        driver = webdriver.Firefox(desired_capabilities=caps, executable_path=r'/Users/QiaoQiao/Documents/SeleniumProject/lib/geckodriver')
        driver.set_window_position(1000,1000)
        driver.set_window_size(480,320)
        driver.get(baseURL)
        # WebDriverWait(driver, delay).until(EC.stalenessOf(p))
        if 'forbes' in baseURL:
            button = '/html/body/div[1]/div[1]/div/div'
            if check_exists_by_xpath(button) == True:
                driver.find_element_by_xpath(button).click()
        WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.TAG_NAME, 'p')))
        sources = driver.find_elements_by_tag_name("p")
        text = ''
        title = driver.find_element_by_tag_name("title").get_attribute("innerText")
        err = ['404 Not Found', 'Page not found - MSN']
        if any(x in title for x in err): 
            print("We have error: " + title)
            driver.quit()
            
        text = text + title + " "
        for source in sources:
            a = source.text
            # a = source.get_attribute("innerText")
            text = text + a + " "
        driver.quit()
        text = text.replace("\n"," ").replace("\t"," ").replace("\f"," ")
        text = text.encode("utf-8","ignore")
        text = text.decode("ascii", "ignore") + " \n"
        print(text)


    except TimeoutException:
        print("============================================")
        print('Loading takes too long!')
        print(URL)

    except StaleElementReferenceException:
        print("=============================================")
        print("Can not locate P in this website!!!")
        print(URL)

URL = sys.argv[1]
scraping_news(URL)