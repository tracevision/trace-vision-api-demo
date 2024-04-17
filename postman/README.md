# Trace Vision API using Postman

This collection and environment JSON file are intended to provide a quick start for using the Trace Vision API from Postman.

## Prerequisites

- **Postman Application**: Ensure that Postman is installed on your computer. If not, download and install it from [Postman's official website](https://www.postman.com/downloads/).

## Getting Started

### 1. Import the Collection and Environment File

#### Import Collection
1. Open Postman.
2. Click on the **Import** button at the top left of the window.
3. Drag and drop the `vision_api_postman_collection.json` file into the target area or use the file picker to locate and select the file.
4. Click **Import** to add the collection to your workspace.

#### Import Environment
1. Repeat the above steps for the `vision_api_postman_environment.json` file, selecting this file instead when you reach the file selection step.

### 2. Configure the Environment Variables

1. In Postman, go to the **Environments** tab on the left sidebar.
2. Select the imported `Vision API` environment.
3. You will see a list of variables like `api_url`, `customer_id`, `api_key`, etc., all initially set to `null`.
4. Fill in the actual values for each variable:
   - `api_url`: The base URL of the Vision API, which is [https://api.tracevision.com/graphql/v1/](https://api.tracevision.com/graphql/v1/)
   - `customer_id`: Your customer ID, found in the [developer portal](https://developer.tracevision.com/)
   - `api_key`: Your API key, found in the [developer portal](https://developer.tracevision.com/)
   - Fill in other variables as necessary, based on your specific use case and API documentation.


Note that you can request access to the API and developer portal [here](https://www.tracevision.com/developer-resources).


### 3. Activate the Environment

- After populating the environment with actual values, click the environment dropdown at the top right corner of Postman (next to the eye icon).
- Select the `Vision API` environment to activate it.

### 4. Use the Collection

- Navigate to the Collections tab and expand the `Vision API` collection.
- Select a request to view or modify it.
- Ensure the request variables (like `{{api_key}}`, `{{customer_id}}`, etc.) are referenced correctly in each request.
- Send a request by clicking the **Send** button.

You may wish to follow our [getting started guide](https://api.tracevision.com/graphql/v1/docs/introduction/getting-started) to create a session, upload your video, and retrieve results.

## Best Practices

- **Secure Handling**: Keep the environment file secure, especially after entering sensitive information such as your API key and customer ID. Avoid sharing this file once you have filled in the variables.
- **Regular Updates**: Keep the environment variables updated according to any changes in the API or security credentials.

## Need Help?

- Full API documentation can be found at [https://api.tracevision.com/graphql/v1/docs/introduction/welcome](https://api.tracevision.com/graphql/v1/docs/introduction/welcome).
- Please reach out to us with your questions. You can find us [here](https://api.tracevision.com/graphql/v1/docs/introduction/contact)

