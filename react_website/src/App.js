import React from 'react';
import { HashRouter, Route, Routes, Navigate } from 'react-router-dom';
import Header from './Header';
import NavBar from './NavBar';
import Home from './Home';
import About from './About';
import Unit from './Unit';
import Page from './Page';
import CustomLink from './CustomLink'
import './App.css';
import { topics } from './Topics';

function App() {
  const subUnitList = topics.map(
    (list, unitIndex) => (
      <Route path = {`/unit${unitIndex+1}`} element={
        <Unit unit={unitIndex+1} subUnitList={
          list.map(
            (title, subUnitIndex) => (
              <li><CustomLink to={`/unit${unitIndex+1}/${subUnitIndex+1}`} className='subunit'>{title}</CustomLink></li>
            )
          )
        }/>
      }>
      </Route>)
    );

  const subUnitRoutes = topics.map(
    (list, unitIndex) => (
      <Routes>{list.map(
        (title, subUnitIndex) => (
          <Route path={`/unit${unitIndex + 1}/${subUnitIndex + 1}`} element={
            <Page unit={unitIndex + 1} index={subUnitIndex + 1} title={title}/>}>
          </Route>
        )
      )}
      </Routes>)
    );

  return (
    <div>
      <HashRouter>
        <Header />
        <NavBar />
        <Routes>
          <Route path="/" element={<Navigate to="/home" />} />
          <Route path="/home" element={<Home />} />
          <Route path="/about" element={<About />} />
          {subUnitList}
        </Routes>
        {subUnitRoutes}
      </HashRouter>
    </div>
  );
}
export default App;
