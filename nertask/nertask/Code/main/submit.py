import requests
import os

CHALLENGE_ID = 2508 
CHALLENGE_PHASE_ID = 5168
SUBMISSION_FILE_PATH = "omitted"
SUBMISSION_COMMENT = "Submitting new xlm-roberta-large baseline"


def submit():
    url = f"https://eval.ai/api/challenges/challenge/{CHALLENGE_ID}/challenge_phase/{CHALLENGE_PHASE_ID}/submission/"
    headers = {
        "Authorization": f"Token {EVALAI_TOKEN}"
    }

    data = {
        "status": "submitting",
        "submission_comment": SUBMISSION_COMMENT
    }

    files = {
        "input_file": open(SUBMISSION_FILE_PATH, 'rb')
    }
    
    try:
        response = requests.post(url, headers=headers, data=data, files=files)
        response.raise_for_status()
        if response.status_code == 201:
            print("Submission successful")
            print(response.json())
        else:
            print("Server response:")
            print(response.json())

    except requests.exceptions.RequestException as e:
        print(f"error occurred during the request: {e}")
        if e.response is not None:
            print("Server response:")
            print(e.response.text)

    finally:
        if 'input_file' in files:
            files['input_file'].close()

if __name__ == "__main__":
    submit()
