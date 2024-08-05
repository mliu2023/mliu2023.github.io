import React from 'react';
import './Resources.css';

function Resources() {
    return (
        <div className="resources">
            <ul>
                <h2>General resources:</h2>
                <li>
                    <a href='https://huggingface.co/models'>HuggingFace</a> - online platform where researchers and AI enthusiasts upload models
                </li>
                <li>
                    <a href='https://www.kaggle.com/'>Kaggle</a> - online community platform of data scientists that hosts challenges and provides a free Jupyter notebook environment
                </li>
                <li>
                    <a href='https://colab.research.google.com/'>Google Colab</a> - free Jupyter notebook environment (preferable to Kaggle)
                </li>
                <li>
                    <a href='https://paperswithcode.com/'>Papers With Code</a> - aggregates state-of-the-art results from research papers along with code
                </li>
                <li>
                    <a href='https://datasetsearch.research.google.com/'>https://datasetsearch.research.google.com/</a> - dataset search
                </li>
                <li>
                    <a href='https://research.google/resources/datasets/'>https://research.google/resources/datasets/</a> - google’s released datasets
                </li>
                <li>
                    <a href='https://course.fast.ai/'>https://course.fast.ai/</a> - Fast AI's comprehensive deep learning course
                </li>
                <li>
                    <a href='https://learn.deeplearning.ai/'>DeepLearning.ai</a> - a set of non-technical short courses for developing AI tools
                </li>
                <li>
                    <a href='https://mango-ambulance-93a.notion.site/ARENA-Virtual-Resources-7934b3cbcfbf4f249acac8842f887a99'>ARENA</a> - exercises on deep learning, transformers, and reinforcement learning with a focus on AI safety.
                </li>
            </ul>
            <br/>
            <ul>
                <h2>Helpful channels:</h2>
                <li><a href='https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&ab_channel=StanfordUniversitySchoolofEngineering'>Stanford CS231</a> (website: http://cs231n.stanford.edu/)</li>
                <li><a href='https://www.youtube.com/watch?v=QDX-1M5Nj7s&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&ab_channel=AlexanderAmini'>MIT 6.S191</a> (website: http://introtodeeplearning.com/)</li>
                <li><a href='https://www.youtube.com/@TheAIEpiphany'>Aleksa Gordić - The AI Epiphany</a></li>
                <li><a href='https://www.youtube.com/@TwoMinutePapers'>TwoMinutePapers</a></li>
                <li><a href='https://www.youtube.com/@YannicKilcher'>Yannic Kilcher</a></li>
                <li><a href='https://www.youtube.com/@outlier4052'>Outlier</a></li>
                <li><a href='https://www.youtube.com/@bycloudAI'>Bycloud</a></li>
            </ul>
        </div>
    );
}
export default Resources;