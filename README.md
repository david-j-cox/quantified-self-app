# quantified-self-app

## Description
The quantified self movement is a movement that seeks to use technology to track and improve our lives. This app is a tool built to help me track data form the many areas of my life where I currently collect data. 

## Installation

To install and set up the quantified-self-app, follow these steps:

1. **Clone the Repository:**
   Open your terminal and run the following command to clone the repository:
   ```bash
   git clone https://github.com/yourusername/quantified-self-app.git
   ```

2. **Navigate to the Project Directory:**
   Change into the project directory:
   ```bash
   cd quantified-self-app
   ```

3. **Install Dependencies:**
   Make sure you have [Node.js](https://nodejs.org/) installed. Then, install the necessary dependencies:
   ```bash
   npm install
   ```

4. **Set Up Environment Variables:**
   Create a `.env` file in the root directory and add your environment variables. For example:
   ```plaintext
   DATABASE_URL=your_database_url
   API_KEY=your_api_key
   ```

5. **Run the Application:**
   Start the application using:
   ```bash
   npm start
   ```

6. **Access the Application:**
   Open your web browser and go to `http://localhost:3000` to access the app.

7. **Optional - Run Tests:**
   If you want to run tests, use:
   ```bash
   npm test
   ```

Make sure to replace placeholders like `yourusername`, `your_database_url`, and `your_api_key` with actual values specific to your setup.

## Usage

Once you have installed and set up the quantified-self-app, you can start using it to track and analyze your data. Here are some examples of how to use the app:

1. **Running the Dashboard:**
   - After starting the application, open your web browser and navigate to `http://localhost:8051`.
   - You will see a dashboard with various tabs such as "OVR Data", "Physical Activity", "Books Read", etc.
   - Click on each tab to explore different visualizations and insights about your data.

2. **Fetching WHOOP Data:**
   - The app can fetch daily data from the WHOOP API. Ensure your WHOOP credentials are set in the `.env` file.
   - Run the script to authenticate and fetch data:
     ```bash
     python Scripts/whoop_data_pull.py
     ```
   - This will store the data in your PostgreSQL database.

3. **Fetching Strava Data:**
   - Similarly, you can fetch activity data from Strava. Ensure your Strava credentials are set in the `.env` file.
   - Run the script to refresh tokens and fetch activities:
     ```bash
     python Scripts/strava_data_pull.py
     ```
   - The data will be appended to the `strava_activities` table in your database.

4. **Customizing Visualizations:**
   - You can modify the scripts in the `Scripts` directory to customize the data visualizations according to your needs.
   - For instance, adjust the plots in `Scripts/app.py` to change how data is displayed on the dashboard.

These examples should help you get started with using the quantified-self-app. Feel free to explore and modify the code to suit your personal tracking needs.