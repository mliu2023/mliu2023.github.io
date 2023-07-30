import React from 'react';
import './About.css';

function About() {
  return (
  <div className="about">
    <section id="section1">
      Hi there! I'm Max Liu, a freshman at Brown University.
      I made this website to help people learn
      about AI/Deep Learning from a compliation of videos and articles.
      Please feel free to email/dm me with any suggestions!
    </section>
    <br />
    <section id="section2">
      Interests: Contest math, AI, golf, video/board games, running
    </section>
    <br />
    <section id="section3">
      Contact information:
      <br />
      <ul>
        <li>Email: maximusliu2004@gmail.com</li>
        <li>Discord: skittlesmurf</li>
        <li>Instagram: max.j.liu (I don't really use this)</li>
      </ul>
    </section>
  </div>);
}
export default About;
