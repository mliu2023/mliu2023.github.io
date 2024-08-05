import React from 'react';
import { HashRouter, Route, Routes, Navigate } from 'react-router-dom';
import Header from './utils/Header';
import NavBar from './utils/NavBar';
import Home from './pages/Home';
import About from './pages/About';
import Resources from './pages/Resources';
import './App.css';
import { subUnitList } from './pages/SubUnitList';
import { subUnitRoutes } from './pages/SubUnitRoutes';

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