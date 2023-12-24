import React, { useEffect, useRef } from 'react';
import About from './About';
import './Home.css';

function Home() {
  const idRef = useRef();

  useEffect(() => {
    let text = [
      'algorithms are too hard.',
      'Attention Is All You Need.',
      'money.',
      'you have leftover GPUs from mining Bitcoin.',
      'drug discovery.',
      'LLMs can write my papers.',
      'hardware is boring.'];
    let timer = setTimeout(() => typeWriter(text), 1000);
    return () => {
      clearTimeout(timer);
      clearTimeout(idRef.current);
    }
  }, []);

  var stringIndex = 0;
  var charIndex = 0;
  let timePerString = 1250;
  let pause = 1000;

  function typeWriter(text) {
    if (document.getElementById("typedtext") != null){
      if (charIndex < text[stringIndex].length) {
        document.getElementById("typedtext").innerHTML += text[stringIndex].charAt(charIndex);
        charIndex++;
        const id = setTimeout(() => { typeWriter(text) }, timePerString / text[stringIndex].length);
        idRef.current = id;
      }
      else {
        const id = setTimeout(() => { typeDeleter(text) }, pause);
        idRef.current = id;
      }
    }
  }
  function typeDeleter(text) {
    if (document.getElementById("typedtext") != null){
      if (charIndex > 0) {
        document.getElementById("typedtext").innerHTML = document.getElementById("typedtext").innerHTML.slice(0, -1);
        charIndex--;
        const id = setTimeout(() => { typeDeleter(text) }, timePerString / text[stringIndex].length);
        idRef.current = id;
      }
      else {
        stringIndex = (stringIndex + 1) % text.length;
        const id = setTimeout(() => { typeWriter(text) }, pause);
        idRef.current = id;
      }
    }
  }

  return (
  <div className="home">
    <div className="tagline">
      <h1>Deep Learning</h1>
      <h5 id="typedtext">Because </h5>
    </div>
  </div>);
}
export default Home;
