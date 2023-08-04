import React from 'react';
import { Link } from 'react-router-dom';
import icon from './images/dl_icon.png';
import './Header.css';

function Header() {
  return (
    <div className="header">
      <Link to="/home"><img src={icon} /></Link>
      <h1>Introduction to Deep Learning</h1>
      <hr />
    </div>
  );
}
export default Header;
