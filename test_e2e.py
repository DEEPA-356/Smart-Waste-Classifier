import requests
import os

def test_end_to_end():
    url = "http://localhost:8080/classify"
    
    # Path to a test image
    test_image_path = "data/val/O/dummy_0.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found at {test_image_path}. Please create one or update the path.")
        return

    print("Testing End-to-End MLOps Waste System...")
    
    with open(test_image_path, "rb") as image_file:
        files = {"file": ("dummy_0.jpg", image_file, "image/jpeg")}
        
        try:
            response = requests.post(url, files=files)
            response.raise_for_status()
            
            data = response.json()
            print("\n✅ End-to-End Flow Successful!")
            print(f"Classification: {data['classification']}")
            print(f"Decomposition Timeline: {data['decomposition_timeline']}")
            print(f"Recycling Instructions: {data['recycling_instructions']}")
            print(f"Upcycling Idea: {data['upcycling_idea']}")
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Test Failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(e.response.text)

if __name__ == "__main__":
    test_end_to_end()
