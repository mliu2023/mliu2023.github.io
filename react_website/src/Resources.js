import React from 'react';
import './Resources.css';

function Resources() {
    return (
        <div className="resources">
            <ul>
                <h2>General resources:</h2>
                <li>
                    <a href='https://huggingface.co/models'>HuggingFace</a> - online platform where researchers and AI enthusiasts upload models that can compete with big tech companies
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
            </ul>
            <br/>
                <h2>Specific resources:</h2>
            <ul>
                <li>
                    <a href='https://pytorch-geometric.readthedocs.io/en/latest/'>PyTorch Geometric</a> - library for graph neural networks
                </li>
                <li>
                    <a href='https://deepchem.io/'>DeepChem</a> - Python library for deep learning on molecular datasets
                </li>
                <li>
                    <a href='https://moody-challenge.physionet.org/'>PhysioNet Challenges</a> - data science challenges related to ECGs and other physiological signals
                </li>
                <li>
                    <a href='https://pharmapsychotic.com/tools.html'>https://pharmapsychotic.com/tools.html</a> - list of AI art tools
                </li>
                <li>
                    <a href='https://gymnasium.farama.org/'>Gymnasium</a> - Python library for creating reinforcement learning environments
                </li>
                <li>
                    <a href='https://developers.google.com/earth-engine/datasets'>https://developers.google.com/earth-engine/datasets</a> - google earth engine
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
            </ul>
        </div>
    );
}
export default Resources;
