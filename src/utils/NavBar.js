import React from 'react';
import CustomLink from './CustomLink';
import './NavBar.css';

function NavBar() {
  return (
    <nav>
      <ul>
        <CustomLink to="/home" className="link">Home</CustomLink>
        <CustomLink to="/about" className="link">About</CustomLink>
        <CustomLink to="/unit1" className="link">Unit 1</CustomLink>
        <CustomLink to="/unit2" className="link">Unit 2</CustomLink>
        <CustomLink to="/unit3" className="link">Unit 3</CustomLink>
        <CustomLink to="/unit4" className="link">Unit 4</CustomLink>
        <CustomLink to="/unit5" className="link">Papers</CustomLink>
        <CustomLink to="/resources" className="link">Resources</CustomLink>
      </ul>
    </nav>
  );
};
export default NavBar;
