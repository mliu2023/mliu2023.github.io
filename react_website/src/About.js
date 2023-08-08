import React from 'react';
import './About.css';

function About() {
  return (
  <div className="about">
    <section id="section1" className="card">
      Hi there! I'm Max Liu, a freshman at Brown University.
      I made this website to help people learn
      about AI/Deep Learning from a compliation of videos and articles.
      Please feel free to email/dm me with any suggestions!
    </section>
    <br />
    <section id="section2" className="card">
      This resource is neither the most efficient nor the most 
      rigorous way to cover deep learning. Instead, it gives 
      non-CS majors a chance to keep up with the latest 
      developments in AI without having to take a class.
    </section>
    <br />
    <section id="section3" className="card">
      Contact information:
      <br />
      <ul>
        <li>Email: maximusliu2004@gmail.com</li>
        <li>Discord: skittlesmurf</li>
      </ul>
    </section>
  </div>);
}
export default About;
