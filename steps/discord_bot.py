import requests
from zenml.steps import step

# This is a private ZenML Discord channel. We will get notified if you use 
# this, but you won't be able to see it. Feel free to create a new Discord 
# [webhook](https://support.discord.com/hc/en-us/articles/228383668-Intro-to-Webhooks) 
# and replace this one!
DISCORD_URL = (
    "https://discord.com/api/webhooks/935835443826659339/Q32jTwmqc"
    "GJAUr-r_J3ouO-zkNQPchJHqTuwJ7dK4wiFzawT2Gu97f6ACt58UKFCxEO9"
)


@step(enable_cache=False)
def discord_alert(
    deployment_decision: bool
) -> None:
    """Send a message to the discord channel to report drift.
    Args:
        deployment_decision: True if drift detected; false otherwise.
    """
    url = DISCORD_URL
    data = {
        "content": "Drift Detected!" if deployment_decision else "No Drift Detected!",
        "username": "Drift Bot",
    }
    result = requests.post(url, json=data)

    try:
        result.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
    else:
        print(
            "Posted to discord successfully, code {}.".format(
                result.status_code
            )
        )
    print("Drift detected" if deployment_decision else "No Drift detected")
    