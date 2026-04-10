from playwright.sync_api import sync_playwright

# Function to take a screenshot and print the page title
def run(playwright):
    browser = playwright.chromium.launch()
    page = browser.new_page()
    page.goto('https://example.com')
    page.screenshot(path='example_screenshot.png')
    print('Page Title:', page.title())
    browser.close()

with sync_playwright() as playwright:
    run(playwright)