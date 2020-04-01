import React, { useEffect, useRef } from "react";
import { transition, min, max, select, line, curveLinear, curveCardinal,scalePoint,axisLeft,scaleLinear,axisBottom,schemeCategory10 } from "d3";
import useResizeObserver from "./useResizeObserver";

function LineChart({data, progressLine}) {

    const svgRef = useRef();
    const wrapperRef = useRef();
    const dimensions = useResizeObserver(wrapperRef);

    

    // will be called initially and on every data change
    useEffect(() => {

    var lines = data.map(d => {return d.probabilities});
    
    var legendData = [];
    data.map((d,index) => {
      const maxVal = max(d.probabilities);
      
      if (maxVal > 0.15) {
        const cat = d.category;
        legendData.push({'category': cat, 'color': schemeCategory10[index % 10]})
      }
    });

    const maxYVal = max(lines.map(d => max(d)));
    
    // const rectCoordinate = [progressLine];
    const svg = select(svgRef.current);
    // const svgContent = svg.select(".content");
    const { width, height } =
      dimensions || wrapperRef.current.getBoundingClientRect();

    const xScale = scaleLinear()
      .domain([0,lines[0].length])
      .range([0, width]);

    const yScale = scaleLinear()
      .domain([0, maxYVal])
      .range([height,0]);

    const lineGenerator = line()
      .x((d,index) => xScale(index))
      .y(d => yScale(d))
      .curve(curveCardinal);


      
      // Legend
    var legend = svg.select(".legend").selectAll('g')
    .data(legendData)
    .enter()
    .append('g')
    .attr('class', 'legend');
    
    legend.append('rect')
      .attr('x', width - 40)
      .attr('y', function(d, i) {
        return i * 20 + 9;
      })
      .attr('width', 10)
      .attr('height', 10)
      .style('fill',d => d.color);

    legend.append('text')
      .attr('x', width - 8)
      .attr('y', function(d, i) {
        return (i * 20) + 18;
      })
      .text(function(d) {
        return d.category;
      });

    svg
      .selectAll(".myLine")
      .data(lines) 
      .join("path")
      .attr("class", "myLine")
      .attr("stroke", function(d,i) {return schemeCategory10[i % 10]})
      .attr("fill", "none")
      .attr("d", d => lineGenerator(d));


      
    // axes
    const xAxis = axisBottom(xScale);
    svg
      .select(".x-axis")
      .attr("transform", `translate(0, ${height})`)
      .call(xAxis);

    const yAxis = axisLeft(yScale);
    svg.select(".y-axis").call(yAxis);

  }, [data]);

    useEffect(() => {

      var lines = data.map(d => {return d.probabilities});
    
    var legendData = [];
    data.map((d,index) => {
      const maxVal = max(d.probabilities);
      
      if (maxVal > 0.15) {
        const cat = d.category;
        legendData.push({'category': cat, 'color': schemeCategory10[index % 10]})
      }
    });

    const maxYVal = max(lines.map(d => max(d)));

      const svg = select(svgRef.current);
    // const svgContent = svg.select(".content");
    const { width, height } =
      dimensions || wrapperRef.current.getBoundingClientRect();

      const xScale = scaleLinear()
      .domain([0,lines[0].length])
      .range([0, width]);

      const yScale = scaleLinear()
      .domain([0, maxYVal])
      .range([height,0]);

      const values = [progressLine];

      var progress = svg.select(".progressBar").selectAll("rect")
      .data(values)

      progress.exit().remove();

      progress.attr('x', d => {console.log(d) ; return xScale(d)})

      progress.enter()
      .append('rect')
      .attr('x', d => {console.log(d) ; return xScale(d)})
      .attr('y', yScale(1.0))
      .attr('width', 5)
      .attr('height', height)
      .style('fill', 'red')
      .style('opacity', 0.8);

    }, [progressLine]);

  return (
    <React.Fragment>
     <div ref={wrapperRef} style={{ marginBottom: "2rem" }}>
      <svg ref={svgRef}>
          <g className="x-axis" />
          <g className="y-axis" />
          <g className="legend" />
          <g className="progressBar" />
        </svg>
      </div>
    </React.Fragment>
  );
}

export default LineChart;