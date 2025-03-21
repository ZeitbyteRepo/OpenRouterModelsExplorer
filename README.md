# OpenRouter Models Explorer

A Streamlit application to explore and analyze models from OpenRouter API, with features for filtering, sorting, and visualizing model data.

## Features

- View all available models with detailed information
- Filter models by moderation status (All/Moderated/Unmoderated)
- Sort and filter by any column
- Interactive visualizations:
  - Model distribution by modality
  - Context length distribution
  - Moderation status distribution
  - Correlation matrix for numeric features
- Export data to CSV or JSON formats
- Customizable column visibility

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/openrouter-explorer.git
cd openrouter-explorer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.streamlit/secrets.toml` file with your OpenRouter API key:
```toml
OPENROUTER_API_KEY = "your-api-key-here"
```

## Usage

Run the Streamlit app locally:
```bash
streamlit run openrouter_streamlit.py
```

## Deployment

This app can be deployed on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Add your OpenRouter API key in the Streamlit Cloud secrets management
5. Deploy!

## Contributing

Feel free to open issues or submit pull requests for any improvements.

## License

MIT License