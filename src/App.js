import React from 'react';
import { HashRouter, Route, Routes, Navigate } from 'react-router-dom';
import Header from './Header';
import NavBar from './NavBar';
import Home from './Home';
import About from './About';
import Resources from './Resources';
import './App.css';
import { subUnitList } from './SubUnitList';
import { subUnitRoutes } from './SubUnitRoutes';

function App() {
  return (
    <div className="app">
      <HashRouter>
        <Header />
        <NavBar />
        <Routes>
          <Route path="/" element={<Navigate to="/home" />} />
          <Route path="/home" element={<Home />} />
          <Route path="/about" element={<About />} />
          <Route path="/resources" element={<Resources />} />
          {subUnitList}
        </Routes>
        {subUnitRoutes}
      </HashRouter>
    </div>
  );
}
export default App;